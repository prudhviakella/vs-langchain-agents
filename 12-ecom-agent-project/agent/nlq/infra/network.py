"""Network path from the NLQ Lambda to RDS — inside the default VPC, no NAT gateway.

    Lambda (ecom-nlq-lambda-sg, default VPC subnets)
        ├─ 5432 ──► RDS            rule on the RDS security group: source = the Lambda SG (not an IP)
        └─ 443  ──► Secrets Manager interface endpoint (private DNS; ecom-nlq-endpoint-sg)
                    — reads the read-only user's password without leaving the VPC

WHY AN ENDPOINT, NOT A NAT GATEWAY
    The Lambda needs exactly one AWS API (Secrets Manager). An interface
    endpoint is about USD 7 a month; a NAT gateway about USD 32 plus data.

WHAT THIS DOES NOT DO
    It does not change the RDS instance, and it leaves your laptop's IP rules alone.
    Safe to re-run: every resource is looked up before it is created.
"""
import boto3

LAMBDA_SG = "ecom-nlq-lambda-sg"
ENDPOINT_SG = "ecom-nlq-endpoint-sg"

ec2 = boto3.client("ec2")


def _sg(name: str, vpc_id: str, description: str) -> str:
    found = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]},
                                                  {"Name": "vpc-id", "Values": [vpc_id]}])["SecurityGroups"]
    if found:
        return found[0]["GroupId"]
    sg_id = ec2.create_security_group(GroupName=name, Description=description, VpcId=vpc_id)["GroupId"]
    print(f"  security group {name} created ({sg_id})")
    return sg_id


def _allow_from_group(target_sg: str, source_sg: str, port: int, note: str) -> None:
    perms = ec2.describe_security_groups(GroupIds=[target_sg])["SecurityGroups"][0].get("IpPermissions", [])
    if any(p.get("FromPort") == port and any(g.get("GroupId") == source_sg for g in p.get("UserIdGroupPairs", []))
           for p in perms):
        return
    ec2.authorize_security_group_ingress(GroupId=target_sg, IpPermissions=[{
        "IpProtocol": "tcp", "FromPort": port, "ToPort": port,
        "UserIdGroupPairs": [{"GroupId": source_sg, "Description": note}]}])
    print(f"  {target_sg} now admits {source_sg} on {port}")


def ensure(rds_security_group: str, region: str) -> dict:
    """{vpc_id, subnet_ids, lambda_sg} — the Lambda's network, ready to reach RDS."""
    vpc_id = ec2.describe_security_groups(GroupIds=[rds_security_group])["SecurityGroups"][0]["VpcId"]
    subnets = ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])["Subnets"]
    subnet_ids = [s["SubnetId"] for s in subnets]
    lambda_sg = _sg(LAMBDA_SG, vpc_id, "ecom NLQ SQL Lambda - outbound only")
    endpoint_sg = _sg(ENDPOINT_SG, vpc_id, "ecom NLQ - Secrets Manager endpoint")
    _allow_from_group(rds_security_group, lambda_sg, 5432, "ecom-nlq SQL Lambda")
    _allow_from_group(endpoint_sg, lambda_sg, 443, "ecom-nlq SQL Lambda")

    service = f"com.amazonaws.{region}.secretsmanager"
    existing = ec2.describe_vpc_endpoints(Filters=[{"Name": "vpc-id", "Values": [vpc_id]},
                                                   {"Name": "service-name", "Values": [service]}])["VpcEndpoints"]
    live = [e for e in existing if e["State"].lower() not in ("deleted", "deleting", "failed", "rejected")]
    if live:
        print(f"  Secrets Manager endpoint exists ({live[0]['VpcEndpointId']})")
    else:
        # one subnet per availability zone: an interface endpoint takes at most one per zone
        per_zone = list({s["AvailabilityZone"]: s["SubnetId"] for s in subnets}.values())
        ep = ec2.create_vpc_endpoint(VpcEndpointType="Interface", VpcId=vpc_id, ServiceName=service,
                                     SubnetIds=per_zone, SecurityGroupIds=[endpoint_sg], PrivateDnsEnabled=True)
        print(f"  Secrets Manager endpoint created ({ep['VpcEndpoint']['VpcEndpointId']})")
    return {"vpc_id": vpc_id, "subnet_ids": subnet_ids, "lambda_sg": lambda_sg}
