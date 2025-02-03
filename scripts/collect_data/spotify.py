"""
Spotify API Wrapper Module

This module provides a wrapper around the Spotify API for retrieving song information and genres.
It implements a rotation system for API keys to handle rate limiting effectively.

The module uses multiple API keys and automatically rotates through them when requests timeout
or fail, ensuring continuous operation even under heavy usage.

Dependencies:
    - requests
    - spotipy
    - json
    - threading
    - os
    - random

Note:
    API keys should be kept secure and not shared publicly.
"""

# this script is a wrapper around the spotify api. it is used to get song ids from the spotify api and to get the genres of songs. The api has pretty harsh rate limits, so we need to be careful not to overload it. to do this, we have a list of api keys, and we cycle through them to avoid rate limiting.

import requests
import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import threading
import os
import random
from typing import Union, List, Tuple

api_keys = [
    {"client_id": "5831e85b191f48f193b578b511c2715b", "client_secret": "e674a61d5b414759a091a8498de991dc"},
    {"client_id": "224b2764de1542949945df422b0455fe", "client_secret": "1546e21031284cc18b85524e70aea73e"},
    {"client_id": "3e950b4c5afe4605be22fa728511466b", "client_secret": "3d582ee59c944e2b888f987221428b2d"},
    {"client_id": "0c2ab5a5ac9d42e187224dd76d1eaa84", "client_secret": "8a7c9dce47714b0eb4db853b11c77cbb"},
    {"client_id": "023e420f37684dac97f08f30d770366f", "client_secret": "7c5d7ddf3bc04027a0431ed66133501d"},
    {"client_id": "b32b082cea4943a896d919038bb90d2e", "client_secret": "c7ddc118da8940babb369ee27e6bbfc9"},
    {"client_id": "15f12336472b4452aaf1a5f078125415", "client_secret": "e0c006cfe78a4c92acb29aeff7b12d18"},
    {"client_id": "6da0eb5e4991455e9811c0c08d9dea66", "client_secret": "26484a13d8ec45d893c8275fc234a2ab"},
    {"client_id": "de3fed019eef41bbb7704d353af81b93", "client_secret": "2c1f71f544a24a4890a6ffb1bbdcb64f"},
    {"client_id": "f320ca921d184e36acba88323cdc5ea3", "client_secret": "6b4e10ecebf84543b2a9382c49508e00"},
    {"client_id": "861df6d0ee264730a4e3009fe4dcd414", "client_secret": "8158a0d8f8f94485b8b7205cc4983e8e"},
    {"client_id": "bc31344a07b24d20b7c4941d465fb822", "client_secret": "5597b972322245088236af63d78597e0"},
    {"client_id": "207cba57ab4148989478b0d30ef1a671", "client_secret": "0ac5250f12944f6b963b305edb2ad562"},
    {"client_id": "bb28fd541cc741a1bc9930a55a4a1798", "client_secret": "f8636472395846c7bc06c5141e8fc7ce"},
    {"client_id": "eb43b563891541e2b3b897f1f32522f1", "client_secret": "91e6ee3f00a3453990e90e3d50f13d89"},
    {"client_id": "dbb20786e5e74f459a219c89cf52b1c5", "client_secret": "ebe97d93d2e64408aeece4971eaefc56"},
    {"client_id": "94ff3e1b64a34716929732f08c12add1", "client_secret": "23b90954ba504341aeacf9176937e940"},
    {"client_id": "6c466cb57b66469bb8da5a84cd756272", "client_secret": "f813e7c65a6a4cd4b1d45d11138cade7"},
    {"client_id": "7d3e9429370843e18c799536ea34c65a", "client_secret": "c3bc060ce99146f48b45a7ea444bb608"},
    {"client_id": "554485873692469d81f7df68b60dbfd6", "client_secret": "314c2d1005e44ea09298edf976684840"},
    {"client_id": "beefd5e410df41c39e21185d54ed7c27", "client_secret": "578ab52cbb3b4a7abfeef28200f5198e"},
    {"client_id": "8cacae76dbe9421787ced13c5f5ba82a", "client_secret": "a6e8ffb461b74d9dbc2ef9586fc66143"},
    {"client_id": "5a13f398c0ff4c78a3769430a20f59b5", "client_secret": "bc3d1936bc20476b99d614bdf5f2162b"},
    {"client_id": "ecb30772ae83409e892172f932c2f324", "client_secret": "6d88bc2b8fce4685bae71b70d2bb275c"},
    {"client_id": "e30b6c769e684ba68f04a3ca1a6c25dc", "client_secret": "898ede83cbaa459d9bc2091538133094"},
    {"client_id": "1885147300844bbd9e354429db8927e4", "client_secret": "7f07b62dc3704235a6dc176085114a88"},
    {"client_id": "70805338a8f14b58a1bfe401d7f9661d", "client_secret": "54a64b8405254340a042fe10af104152"},
    {"client_id": "358f737f62dc4729b9cda19661fef156", "client_secret": "ddd8746a826543b2834e9dce732d6b7d"},
    {"client_id": "1fc3247139cc4b26b1ecae34c2749cef", "client_secret": "16b0cf12e11444cda0d08488df5f081a"},
    {"client_id": "95ecf868a34f48809cb0c830bd19168c", "client_secret": "78dedceb98dd4cb1834f5282deaeb05e"},
    {"client_id": "6aa3f6a867fd4f5c8e2a3c45bc77ecd3", "client_secret": "0c6e37c9a9de4746b968b2030a92e2ba"},
    {"client_id": "743c6f7f25214905b376671dc341048c", "client_secret": "7bf3ac55dae04597972ae004fd113bb0"},
]


def init_spotipy(client_id: str, client_secret: str) -> spotipy.Spotify:
    """
    Initialize a Spotify client with the given credentials.

    Args:
        client_id (str): Spotify API client ID
        client_secret (str): Spotify API client secret

    Returns:
        spotipy.Spotify: Initialized Spotify client object
    """
    credentials = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=credentials)

class SpotifyManager:
    """
    Manages Spotify API interactions with automatic key rotation and error handling.

    This class handles API requests, timeout management, and key rotation to ensure
    reliable access to the Spotify API even under rate limiting conditions.
    """

    def __init__(self):
        """Initialize the SpotifyManager with the first API key."""
        self.current_key_index = 0
        self.sp = self._get_next_spotipy_client()

    def _get_next_spotipy_client(self) -> spotipy.Spotify:
        """
        Rotate to the next API key and initialize a new Spotify client.

        Returns:
            spotipy.Spotify: New Spotify client with the next API key
        """
        cache_file_path = '.cache'
        if os.path.exists(cache_file_path):
            os.remove(cache_file_path)
            print(f"Cache file '{cache_file_path}' has been deleted.")
        
        self.current_key_index = (self.current_key_index + 1) % len(api_keys)
        next_key = api_keys[self.current_key_index]
        return init_spotipy(next_key['client_id'], next_key['client_secret'])

    def _general_request_func(self, sp: spotipy.Spotify, params: dict, 
                            result_container: dict, error_container: dict) -> None:
        """
        Execute a generic Spotify API request.

        Args:
            sp (spotipy.Spotify): Spotify client instance
            params (dict): Request parameters including method, args, and kwargs
            result_container (dict): Dictionary to store the API response
            error_container (dict): Dictionary to store any errors that occur
        """
        try:
            response = getattr(sp, params['method'])(*params['args'], **params['kwargs'])
            result_container['response'] = response
        except Exception as e:
            error_container['error'] = e

    def timed_request(self, request_params: dict, timeout_seconds: int = 10) -> Union[dict, str, None]:
        """
        Execute a Spotify API request with timeout handling.

        Args:
            request_params (dict): Parameters for the API request
            timeout_seconds (int, optional): Maximum time to wait for response. Defaults to 10.

        Returns:
            Union[dict, str, None]: API response, 'timed_out' string, or None if error occurs
        """
        result_container = {}
        error_container = {}

        request_thread = threading.Thread(
            target=self._general_request_func, 
            args=(self.sp, request_params, result_container, error_container)
        )
        request_thread.start()
        request_thread.join(timeout=timeout_seconds)

        if request_thread.is_alive():
            print("Request timed out.")
            self.sp = self._get_next_spotipy_client()
            return 'timed_out'
        elif 'error' in error_container:
            print(f"An error occurred: {error_container['error']}")
            return None
        else:
            return result_container['response']

    def get_genres_for_songs_batch(self, song_ids: List[str]) -> Tuple[dict, dict, dict]:
        """
        Retrieve genres for a batch of songs by looking up their artists.

        This method handles batching of requests to stay within API limits and retrieves
        both track and artist information to compile a complete genre listing for each song.

        Args:
            song_ids (List[str]): List of Spotify track IDs

        Returns:
            Tuple[dict, dict, dict]: Contains three dictionaries:
                - song_genres: Mapping of song IDs to their genres
                - tracks_data: Raw track information from Spotify
                - artists_data: Raw artist information from Spotify

        Example:
            >>> manager = SpotifyManager()
            >>> song_genres, tracks, artists = manager.get_genres_for_songs_batch(['track_id1', 'track_id2'])
        """
        try:
            all_artist_ids = set()
            tracks_by_id = {}
            tracks_data = {'tracks': []}
            artists_data = {'artists': []}
            
            for i in range(0, len(song_ids), 50):
                batch_ids = song_ids[i:i + 50]
                tracks_request_params = {
                    'method': 'tracks',
                    'args': [batch_ids],
                    'kwargs': {}
                }
                tracks_response = self.timed_request(tracks_request_params)
                
                if tracks_response == 'timed_out':
                    return self.get_genres_for_songs_batch(song_ids)
                
                tracks_data['tracks'].extend(tracks_response['tracks'])
                for track in tracks_response['tracks']:
                    if track is None:
                        continue
                    tracks_by_id[track['id']] = track
                    all_artist_ids.update(artist['id'] for artist in track['artists'])
            
            artist_genres = {}
            artist_ids_list = list(all_artist_ids)
            
            for i in range(0, len(artist_ids_list), 50):
                batch_artist_ids = artist_ids_list[i:i + 50]
                artists_request_params = {
                    'method': 'artists',
                    'args': [batch_artist_ids],
                    'kwargs': {}
                }
                artists_response = self.timed_request(artists_request_params)
                
                if artists_response == 'timed_out':
                    return self.get_genres_for_songs_batch(song_ids)
                
                artists_data['artists'].extend(artists_response['artists'])
                for artist in artists_response['artists']:
                    if artist is None:
                        continue
                    artist_genres[artist['id']] = artist['genres']
            
            song_genres = {}
            for song_id in song_ids:
                if song_id not in tracks_by_id:
                    continue
                
                track = tracks_by_id[song_id]
                genres = set()
                for artist in track['artists']:
                    genres.update(artist_genres.get(artist['id'], []))
                song_genres[song_id] = list(genres)
            
            return song_genres, tracks_data, artists_data
            
        except Exception as e:
            print(f"Error getting genres for song batch: {e}")
            return {}, {'tracks': []}, {'artists': []}
