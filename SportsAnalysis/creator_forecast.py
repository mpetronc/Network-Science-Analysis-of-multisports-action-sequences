from pathlib import Path
import argparse
from prospective_absence import main as forecast

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--creator-forecast',action='store_true');p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'nba_passing_output');p.add_argument('--data-dir',type=Path);p.add_argument('--skip-network-plots',action='store_true');a=p.parse_args(argv);data=a.data_dir or a.output_dir/'creator_dependency';args=['--output-dir',str(a.output_dir),'--star-source',str(data/'selected_stars.csv'),'--star-definition','2024-25 primary creator: highest potential assists per36 with at least 1000 tracked minutes','--destination',str(data/'prospective_absence')]
 if a.skip_network_plots:args.append('--skip-network-plots')
 return forecast(args)
if __name__=='__main__':raise SystemExit(main())
