"""Run isolated SQL Server Developer container and restore official WWI backup.

Uses a task-local secret file, no published ports and a persistent task volume.
Never changes an existing database/container outside the kg-erp namespace.
"""
import json
from pathlib import Path
import secrets
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = "kg-erp-sql"
DB = "KGErpWWI"

def run(args, **kw):
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", **kw)
    if result.returncode:
        raise RuntimeError((result.stderr + result.stdout).strip())
    return result.stdout

def query(sql, json_result=False):
    args = ["docker", "exec", NAME, "sh", "-c",
            'SQLCMDPASSWORD="$MSSQL_SA_PASSWORD" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -C -b "$@"',
            "sqlcmd", "-Q", "SET NOCOUNT ON; " + sql]
    args += ["-y", "0", "-w", "65535"] if json_result else ["-h", "-1", "-W", "-s", "|"]
    output = run(args)
    output = '\n'.join(line for line in output.splitlines() if not line.startswith("Changed database context to"))
    if json_result:
        payload = "".join(output.splitlines())
        return json.loads(payload[payload.index("["):payload.rindex("]") + 1])
    return output

def main():
    secret_dir = ROOT / ".secrets"
    secret_dir.mkdir(exist_ok=True)
    env = secret_dir / "sql.env"
    if not env.exists():
        env.write_text("ACCEPT_EULA=Y\nMSSQL_PID=Developer\nMSSQL_SA_PASSWORD=" + secrets.token_urlsafe(36) + "aA1!\n", encoding="utf-8")
    names = run(["docker", "ps", "-a", "--format", "{{.Names}}"] ).splitlines()
    if NAME not in names:
        run(["docker", "run", "-d", "--name", NAME, "--label", "project=kg-erp",
             "--env-file", str(env), "--memory", "6g", "--cpus", "4", "-v", "kg-erp-sql-data:/var/opt/mssql",
             "mcr.microsoft.com/mssql/server:2022-latest"])
    else:
        run(["docker", "start", NAME])
    for _ in range(90):
        try:
            query("SELECT 1")
            break
        except RuntimeError:
            time.sleep(2)
    else:
        raise RuntimeError("SQL Server did not become ready")
    if "1" in query(f"SELECT COUNT(*) FROM sys.databases WHERE name='{DB}'").strip():
        print("Existing project WWI database retained")
        return
    backup = ROOT / "data/raw/WideWorldImporters-Standard.bak"
    run(["docker", "cp", str(backup), f"{NAME}:/var/opt/mssql/data/wwi.bak"])
    files = query("RESTORE FILELISTONLY FROM DISK='/var/opt/mssql/data/wwi.bak'")
    moves = []
    for line in files.splitlines():
        fields = [x.strip() for x in line.split("|")]
        if len(fields) > 3 and fields[2] in {"D", "L", "S"}:
            logical, _, kind = fields[:3]
            filename = "kgerp_" + "".join(c if c.isalnum() else "_" for c in logical)
            filename += ".ldf" if kind == "L" else ".mdf" if kind == "D" else ""
            moves.append(f"MOVE '{logical}' TO '/var/opt/mssql/data/{filename}'")
    if not moves:
        raise RuntimeError("Could not parse backup file list")
    query(f"RESTORE DATABASE [{DB}] FROM DISK='/var/opt/mssql/data/wwi.bak' WITH " + ", ".join(moves))
    (ROOT / "data/manifests/sql_server.json").write_text(json.dumps({
        "container": NAME, "database": DB,
        "server": query("SELECT @@VERSION").strip(),
        "image": run(["docker", "inspect", "--format", "{{.Image}}", NAME]).strip(),
    }, indent=2), encoding="utf-8")
    print("Restored official WWI Standard v1.0 into isolated project container")

if __name__ == "__main__":
    main()
