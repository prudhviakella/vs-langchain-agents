"""Default VPC networking.

Uses the account's default VPC rather than creating one. A dedicated VPC is
the more correct answer for a production deployment with real network
isolation requirements, but it is also several hundred lines this project
does not need to own — the default VPC exists in every account unless
someone has deliberately removed it, and reusing it is what keeps deploy.py
a single command rather than a networking project of its own.
"""

import boto3

from . import config

ec2 = boto3.client("ec2", region_name=config.REGION)


def default_network() -> tuple[list[str], str]:
    """Subnet ids and a security group id from the default VPC.

    Raises with a clear message rather than a bare KeyError if there is no
    default VPC — that is a real, if uncommon, account state, and the fix
    (create one, or point this at an existing VPC) is worth stating rather
    than making someone read a traceback to find it.
    """
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    if not vpcs["Vpcs"]:
        raise RuntimeError(
            "no default VPC in this account/region. Either create one "
            "(aws ec2 create-default-vpc) or edit infra/network.py to "
            "point at an existing VPC's subnets and security group.")
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    subnets = ec2.describe_subnets(
        Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])
    subnet_ids = [s["SubnetId"] for s in subnets["Subnets"]]

    sg_name = f"{config.PROJECT}-sg"
    existing = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [sg_name]},
                {"Name": "vpc-id", "Values": [vpc_id]}])
    if existing["SecurityGroups"]:
        sg_id = existing["SecurityGroups"][0]["GroupId"]
    else:
        sg = ec2.create_security_group(
            GroupName=sg_name, Description="RAG pipeline Batch instances",
            VpcId=vpc_id)
        sg_id = sg["GroupId"]
        # Outbound only. Workers call S3, DynamoDB, OpenAI, Pinecone, and
        # Neo4j Aura, all outbound; nothing ever needs to reach a worker on
        # an inbound port. The default security group already allows all
        # outbound traffic, so nothing further is added here.

    return subnet_ids, sg_id
