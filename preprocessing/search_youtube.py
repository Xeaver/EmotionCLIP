from youtubesearchpython import *
import json
from tqdm import tqdm 

def search_playlist(search_term = 'tv shows', out_file=None, test=False):
    if test: print('Only one video link will be stored because "test" is enabled.')
    tv_shows_meta = list()
    search = PlaylistsSearch(search_term)
    while True:
        result = search.result()
        search.next()
        if result and len(result['result'])==0:
            break
        tv_shows_meta.extend(result['result'])
        if test:
            print(len(tv_shows_meta)) 
            break
    playlist_tv_shows = list()
    for pllist in tqdm(tv_shows_meta):
        try:
            playlist = Playlist(pllist['link'])
        except Exception as e:
                print(e)
        while playlist.hasMoreVideos:
            try:
                playlist.getNextVideos()
                if test:
                    break
            except Exception as e:
                print(e)
        playlist_tv_shows.extend(playlist.videos)
        if test: break
    print('Found all the videos.',len(playlist_tv_shows))
    ftr = [60,1]
    # videos greater than 2000 seconds are kept
    if test: # test 1 video
        playlist_tv_shows = playlist_tv_shows[1:2]
    else:
        playlist_tv_shows = [p for p in playlist_tv_shows if p['duration'] is not None and (sum([a*b for a,b in zip(ftr, [int(i) for i in p['duration'].split(',')[0].split(":")])])>2000 or len(p['duration'].split(',')[0].split(":"))>2)]
    print('Filtered long videos.',len(playlist_tv_shows))
    if out_file is not None:
        with open(out_file, 'w') as outfile:
            json.dump(playlist_tv_shows, outfile)
    return True


if __name__=='__main__':
    search_playlist(search_term = 'tv shows',out_file = 'youtube_playlist_tv.json',test=True)