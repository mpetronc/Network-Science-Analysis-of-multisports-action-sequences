"""Reproduce the chosen final model and observed regressions from packaged data."""
from pathlib import Path
import argparse,subprocess,sys,os


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--destination',type=Path);p.add_argument('--verify-only',action='store_true');a=p.parse_args();package=a.package_root.resolve();dest=a.destination or package/'reproduced_outputs';dest.mkdir(parents=True,exist_ok=True);code=Path(__file__).resolve().parent
 subprocess.run([sys.executable,'-B',str(code/'verify_initial_forecasts.py'),'--package-root',str(package),'--destination',str(dest/'validation')],check=True)
 if a.verify_only:return
 subprocess.run([sys.executable,'-B',str(code/'previous_season_history.py'),'--output-dir',str(package/'data/nba_passing_output'),'--destination',str(dest/'prediction_model')],check=True)
 subprocess.run([sys.executable,'-B',str(code/'run_regressions.py'),'--package-root',str(package),'--destination',str(dest/'regressions')],check=True)
 print('Reproduced model and regression outputs:',dest)

if __name__=='__main__':main()
