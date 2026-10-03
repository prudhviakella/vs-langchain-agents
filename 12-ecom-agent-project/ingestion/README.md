# E-commerce NLQ — ingestion pipeline

The `ingestion/` part of the project: it prepares the data and loads it into PostgreSQL and
Pinecone, where the agents in `../agent/` will read it.

```
data/raw/*.csv ─► prepare ─► data/clean/*.csv ───────────► deploy / postgres ─► RDS PostgreSQL
   (14 files)       │        DATA_DICTIONARY.md                                  schema ecom · keys · comments
                    │                                                            your IP only · password in Secrets Manager
                    └──────► data/kb_feed_ecom.xlsx ─────────► pinecone ────────► Pinecone index (Titan embeddings)
                             (61 SQL examples, executed)                          nlq-schema · nlq-examples · common
```

Everything runs on your own machine — macOS or Windows — with Python 3.10 or newer.
No Databricks, no Spark, no notebooks.

| Command | Needs | Does |
|---|---|---|
| `python run.py prepare` | nothing online | clean the CSVs, check 48 rules, write the dictionary, build the KB workbook |
| `python run.py deploy` | AWS login | create RDS PostgreSQL (public, **your IP only**) and load it — ~10 min the first time |
| `python run.py postgres` | AWS login | reload the deployed database (or any database set in `.env`) |
| `python run.py pinecone --dry-run` | nothing online | build and check every knowledge-base record; send nothing |
| `python run.py pinecone` | AWS login + Pinecone key | publish the knowledge base to Pinecone |
| `python run.py test` | nothing online | offline tests |
| `python run.py deploy --destroy` | AWS login | delete the RDS instance, its subnet group and security group |

## Where to run the commands — PyCharm's Terminal

```
PyCharm ─ open the project ─► View › Tool Windows › Terminal   (⌥F12 on macOS · Alt+F12 on Windows)
                              starts in the project root, with the project's .venv active
                              cd ingestion                      ← once per Terminal tab
                              (.venv) …/your-project/ingestion %   ← ready
```

- Use the **Terminal** tool window — not the **Python Console**, which only runs Python, not these commands.
- The Terminal opens in the project root (*Settings › Tools › Terminal › Start directory*).
  Every command here runs **inside `ingestion/`**: type `cd ingestion` once in each new
  Terminal tab — or set that Start directory to the `ingestion` folder and skip it.
- It activates the project's virtual environment when *Settings › Tools › Terminal ›
  Activate virtualenv* is ticked (the default). You should see `(.venv)` at the start of the prompt.
- macOS opens zsh; Windows opens PowerShell — the matching section below fits each.

**Outside PyCharm** (Terminal.app, Windows Terminal): go to the project's `ingestion` folder
and activate the environment — macOS `cd <project>/ingestion` then `source ../.venv/bin/activate`;
Windows `cd <project>\ingestion` then `..\.venv\Scripts\Activate.ps1`.

---

## macOS

Commands are for Terminal (zsh or bash).

### 1. Install the tools — once

```bash
brew install python@3.12 awscli libpq      # libpq gives you psql
aws configure                              # access key, secret key, region us-east-1
```

Add `psql` to your PATH: `echo 'export PATH="/opt/homebrew/opt/libpq/bin:$PATH"' >> ~/.zshrc`, then open a new Terminal.

### 2. Set up the project — once

In PyCharm: *Settings › Project › Python Interpreter › Add Interpreter › Add Local
Interpreter › Virtualenv Environment › New*, base interpreter Python 3.12. PyCharm creates
`.venv` in the project root (shared by `ingestion/` and `agent/`) and activates it in every new Terminal tab. Then, in the Terminal:

```bash
cd ingestion
pip install -r requirements.txt
cp .env.example .env                       # then fill in your Pinecone key
```

Already using a shared environment (one `.venv` in a parent folder for several projects)?
Select that one as the project interpreter instead — PyCharm activates whichever is selected.

### 3. Prepare the data

```bash
python run.py prepare
```

Ends with `48 of 48 rules pass` and `✓ done`.

### 4. Create RDS PostgreSQL and load it

```bash
python run.py deploy
```

Ends with `examples: 61 of 61 run on PostgreSQL` and `✓ done`. The address, region and
secret are saved in `postgres/deployment.json` — no password in it.

### 5. Connect with psql

The password is read from Secrets Manager into the connection; it is never shown.

```bash
dep() { python -c "import json; print(json.load(open('postgres/deployment.json'))['$1'])"; }
PGPASSWORD="$(aws secretsmanager get-secret-value --region "$(dep region)" --secret-id "$(dep secret_arn)" \
  --query SecretString --output text | python -c 'import json,sys; print(json.load(sys.stdin)["password"])')" \
psql "host=$(dep host) port=$(dep port) dbname=$(dep database) user=$(dep master_user) sslmode=require"
```

Try `SELECT COUNT(*) FROM ecom.orders;` (50000). Leave with `\q`.

### 6. Publish the knowledge base to Pinecone

Put `PINECONE_API_KEY` in `.env` first.

```bash
python run.py pinecone --dry-run
python run.py pinecone
```

### 7. Reload, rotate, delete

```bash
python run.py postgres                     # reload the data into the deployed database

dep() { python -c "import json; print(json.load(open('postgres/deployment.json'))['$1'])"; }
aws secretsmanager rotate-secret --region "$(dep region)" --secret-id "$(dep secret_arn)"   # new password now

python run.py deploy --destroy             # delete everything in AWS when done
```

---

## Windows

Commands are for **PowerShell** (Windows Terminal or the PowerShell app), not `cmd.exe`.

### 1. Install the tools — once

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Amazon.AWSCLI
aws configure                              # access key, secret key, region us-east-1
```

Close and reopen PowerShell after installing so `python` and `aws` are found.
For `psql`, run the PostgreSQL installer from postgresql.org/download/windows, select only
**Command Line Tools**, and add `C:\Program Files\PostgreSQL\16\bin` to your PATH.

### 2. Set up the project — once

In PyCharm: *Settings › Project › Python Interpreter › Add Interpreter › Add Local
Interpreter › Virtualenv Environment › New*, base interpreter Python 3.12. PyCharm creates
`.venv` in the project root (shared by `ingestion/` and `agent/`) and activates it in every new Terminal tab. Then, in the Terminal:

```powershell
cd ingestion
python -m pip install -r requirements.txt
Copy-Item .env.example .env                # then fill in your Pinecone key
```

If the Terminal shows an error about `Activate.ps1` ("running scripts is disabled on this
system") and no `(.venv)` in the prompt, allow scripts for your user once, then open a new
Terminal tab: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### 3. Prepare the data

```powershell
python run.py prepare
```

Ends with `48 of 48 rules pass` and `✓ done`.

### 4. Create RDS PostgreSQL and load it

```powershell
python run.py deploy
```

Ends with `examples: 61 of 61 run on PostgreSQL` and `✓ done`. The address, region and
secret are saved in `postgres\deployment.json` — no password in it.

### 5. Connect with psql

The password goes from Secrets Manager into a variable for this window only, and is
removed afterwards.

```powershell
$d = Get-Content postgres\deployment.json -Raw | ConvertFrom-Json
$env:PGPASSWORD = (aws secretsmanager get-secret-value --region $d.region --secret-id $d.secret_arn --query SecretString --output text | ConvertFrom-Json).password
psql "host=$($d.host) port=$($d.port) dbname=$($d.database) user=$($d.master_user) sslmode=require"
Remove-Item Env:PGPASSWORD
```

Try `SELECT COUNT(*) FROM ecom.orders;` (50000). Leave with `\q`.

### 6. Publish the knowledge base to Pinecone

Put `PINECONE_API_KEY` in `.env` first.

```powershell
python run.py pinecone --dry-run
python run.py pinecone
```

### 7. Reload, rotate, delete

```powershell
python run.py postgres                     # reload the data into the deployed database

$d = Get-Content postgres\deployment.json -Raw | ConvertFrom-Json
aws secretsmanager rotate-secret --region $d.region --secret-id $d.secret_arn   # new password now

python run.py deploy --destroy             # delete everything in AWS when done
```

---

## Settings — `.env`

| Setting | Needed for | Notes |
|---|---|---|
| `PINECONE_API_KEY` | `pinecone` | or `PINECONE_SECRET_ID` (Secrets Manager id holding `{"api_key": …}`) |
| `PINECONE_INDEX` | `pinecone` | default `ecom-kb`; created on first run |
| `AWS_REGION`, `AWS_PROFILE` | `deploy`, `postgres`, `pinecone` | your usual AWS login; default region `us-east-1` |
| `PGHOST` … `PGPASSWORD` | `postgres` against another database | not needed after `deploy` — it uses `deployment.json` |

`.env` is in `.gitignore`. Never commit it.

---

## Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| `missing packages for run.py …` | the environment lacks a package | check `(.venv)` is in the prompt, then `pip install -r requirements.txt` |
| no `(.venv)` in the Terminal prompt | the Terminal tab opened before the interpreter was set, or *Activate virtualenv* is off | open a new Terminal tab; tick *Settings › Tools › Terminal › Activate virtualenv* |
| `can't open file '…run.py'` | the Terminal is not in `ingestion/` | `cd ingestion` (once per Terminal tab) |
| `cd: no such file or directory: ingestion` | you are already in `ingestion/` (or below the root) | nothing to do — run the command |
| `SyntaxError` on `python run.py …` | typed into the **Python Console** | use the **Terminal** tool window |
| `could not find your public IP (… CERTIFICATE_VERIFY_FAILED …)` | **macOS**, python.org Python without certificates | use the current `deploy.py` (it brings its own); or run the "Install Certificates.command" in Applications › Python 3.12; or pass the IP — macOS: `--allow-cidr "$(curl -s https://checkip.amazonaws.com)/32"`, Windows: `--allow-cidr "$((Invoke-RestMethod https://checkip.amazonaws.com).Trim())/32"` |
| `cannot connect to Postgres … timeout` | your IP changed (new network, VPN) | run `python run.py deploy` again — it adds your new IP |
| `ResourceNotFoundException` for the secret | an id that is not the full ARN | use `secret_arn` from `deployment.json`, as the commands above do |
| `running scripts is disabled on this system` | **Windows** execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `python` opens the Microsoft Store | **Windows** app alias | use `py -3.12`, or turn off the alias in Settings › Apps › Advanced app settings › App execution aliases |
| `psql major version 14, server major version 16` | an older psql client | harmless for queries; `brew install libpq` (macOS) or the version-16 installer (Windows) |
| `?` instead of `━ ✓ ✗` | a console without those characters | cosmetic only — output written to a file looks the same |

---

## Security and cost

- **Who can connect:** only the IPs `deploy` opened, each as a single `/32`. "Publicly accessible" gives the database an internet address; the security group refuses everyone else.
- **The password** is generated by RDS into Secrets Manager and rotates every 7 days. Read it when you need it; never store it. If it was ever pasted somewhere, rotate it (step 7).
- **The NLQ agent** should connect as a read-only user, not `ecom_admin` — the master user can change or drop anything.
- **Cost:** roughly USD 15–20 a month while the instance runs (instance, storage, public IPv4) unless your free tier covers it. `python run.py deploy --destroy` removes it.

---

## Running one script on its own

`run.py` calls these for you; each also runs alone, from `ingestion/`
(macOS shown — on Windows use `\` in the paths):

```bash
python data_prep/fix_data.py data/raw data/clean
python data_prep/validate_data.py data/clean
python data_prep/gen_dictionary.py data/clean data/clean/DATA_DICTIONARY.md
python kb_feed/load_duckdb.py data/clean data/kb_build.duckdb
python kb_feed/build_kb_feed.py --db data/kb_build.duckdb --out data/kb_feed_ecom.xlsx
python postgres/deploy.py --skip-load
python postgres/load_postgres.py --data data/clean --kb-feed data/kb_feed_ecom.xlsx --check-examples
python pinecone_kb/kb_to_pinecone.py data/kb_feed_ecom.xlsx --dry-run
```

## What is where

| Folder | Contents |
|---|---|
| `data/raw/` | the 14 source CSVs |
| `data/clean/` | output: cleaned CSVs, `DATA_DICTIONARY.md`, `changes/` (every changed row) |
| `data_prep/` | `fix_data.py`, `validate_data.py`, `gen_dictionary.py` |
| `kb_feed/` | the knowledge-base content and `build_kb_feed.py` |
| `postgres/` | `schema.sql`, `load_postgres.py`, `deploy.py`, their tests |
| `pinecone_kb/` | `kb_to_pinecone.py` and its tests |
