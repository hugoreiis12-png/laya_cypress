from pathlib import Path
import os, sys
home=Path(__file__).resolve().parents[1]
python=home/"runtime-env"/("Scripts/python.exe" if os.name=="nt" else "bin/python")
if python.exists() and Path(sys.prefix).resolve() != python.parent.parent.resolve():
    if os.name=="nt":
        # no Windows os.execv devolve o controle antes do filho terminar e a saida se perde
        import subprocess
        raise SystemExit(subprocess.call([str(python),*sys.argv]))
    os.execv(str(python),[str(python),*sys.argv])
sys.path.insert(0,str(home/"lib"))
from itsfine.cli import main
raise SystemExit(main())
