"""Isolate nba_api's global HTTP session for each download worker."""
import threading
import requests
from nba_api.stats.library.http import NBAStatsHTTP

def install():
 if getattr(NBAStatsHTTP,'_hub_thread_local',False):return
 local=threading.local()
 def get_session(cls):
  if not hasattr(local,'session'):local.session=requests.Session()
  return local.session
 def set_session(cls,session):local.session=session
 NBAStatsHTTP.get_session=classmethod(get_session)
 NBAStatsHTTP.set_session=classmethod(set_session)
 NBAStatsHTTP._hub_thread_local=True
