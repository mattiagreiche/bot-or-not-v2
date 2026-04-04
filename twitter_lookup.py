"""
twitter_lookup.py

Queries getxapi.com for each unique username found in the dataset JSONs,
computes account existence and profile-match features, and saves results to
raw_data/twitter_lookup.csv.

Set env var GETXAPI_KEY before running.
Supports resuming: already-processed usernames are skipped.
"""

import concurrent.futures
import csv
import json
import os
import threading
import time
from difflib import SequenceMatcher
from dotenv import load_dotenv

import requests

from botornot.config import DIR_RAW, INFERENCE_POST_FILES, TRAINING_POST_FILES

load_dotenv()

API_URL = "https://api.getxapi.com/twitter/user/info"
OUTPUT_PATH = os.path.join(DIR_RAW, "twitter_lookup.csv")
CONCURRENCY = 10

COLUMNS = [
    "username",
    "account_exists",
    "username_match",
    "name_match",
    "description_match",
    "description_partial_match",
    "location_match",
]

# --- thread-safe shared state ---
_thread_local = threading.local()
_write_lock = threading.Lock()
_rate_lock = threading.Lock()
_rate_limited_until = 0.0  # monotonic timestamp


def get_session(api_key):
    """One requests.Session per thread."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers["Authorization"] = f"Bearer {api_key}"
        _thread_local.session = s
    return _thread_local.session


def wait_if_rate_limited():
    """Block until the global rate-limit window has passed."""
    while True:
        with _rate_lock:
            wait = _rate_limited_until - time.monotonic()
        if wait <= 0:
            return
        time.sleep(min(wait, 1.0))


def set_rate_limit(backoff_s):
    global _rate_limited_until
    with _rate_lock:
        _rate_limited_until = max(_rate_limited_until, time.monotonic() + backoff_s)


# --- helpers ---

def load_users_from_files(file_list):
    """Return {username: {name, description, location}} from all dataset JSONs."""
    users = {}
    for filepath in file_list:
        if not os.path.exists(filepath):
            print(f"  Skipping missing file: {filepath}")
            continue
        with open(filepath) as f:
            data = json.load(f)
        for u in data.get("users", []):
            uname = (u.get("username") or "").strip()
            if uname and uname not in users:
                users[uname] = {
                    "name": (u.get("name") or "").strip(),
                    "description": (u.get("description") or "").strip(),
                    "location": (u.get("location") or "").strip(),
                }
    return users


def fuzzy_similarity(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def lookup_user(api_key, username, local):
    """Fetch user from getxapi, retrying on 429. Returns a feature dict."""
    error_row = dict.fromkeys(COLUMNS, -1)
    error_row["username"] = username

    not_found_row = dict.fromkeys(COLUMNS, -1)
    not_found_row["username"] = username
    not_found_row["account_exists"] = 0

    session = get_session(api_key)
    backoff = 60

    while True:
        wait_if_rate_limited()
        try:
            resp = session.get(API_URL, params={"userName": username}, timeout=15)

            if resp.status_code == 429:
                set_rate_limit(backoff)
                backoff = min(backoff * 2, 300)
                continue

            if resp.status_code == 404:
                return not_found_row

            if resp.status_code != 200:
                print(f"  Unexpected HTTP {resp.status_code} for @{username}")
                return error_row

            body = resp.json()
            if "error" in body or body.get("status") != "success":
                return not_found_row

            data = body.get("data") or {}
            api_username = (data.get("userName") or "").strip()
            api_name = (data.get("name") or "").strip()
            api_desc = (data.get("description") or "").strip()
            api_loc = (data.get("location") or "").strip()

            return {
                "username": username,
                "account_exists": 1,
                "username_match": int(api_username.lower() == username.lower()),
                "name_match": int(api_name.lower() == local["name"].lower()),
                "description_match": int(api_desc.lower() == local["description"].lower()),
                "description_partial_match": round(fuzzy_similarity(api_desc, local["description"]), 4),
                "location_match": int(api_loc.lower() == local["location"].lower()),
            }

        except Exception as e:
            print(f"  Exception for @{username}: {e}")
            return error_row


def main():
    api_key = os.environ.get("GETXAPI_KEY")
    if not api_key:
        raise EnvironmentError("GETXAPI_KEY environment variable is not set.")

    all_files = TRAINING_POST_FILES + INFERENCE_POST_FILES
    print(f"Loading users from {len(all_files)} dataset files...")
    users = load_users_from_files(all_files)
    print(f"Found {len(users)} unique usernames.")

    done = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["username"])
        print(f"Resuming: {len(done)} already done, {len(users) - len(done)} remaining.")

    todo = [(uname, local) for uname, local in users.items() if uname not in done]
    if not todo:
        print("Nothing to do.")
        return

    total = len(todo)
    completed = 0

    write_header = not os.path.exists(OUTPUT_PATH)
    with open(OUTPUT_PATH, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {
                executor.submit(lookup_user, api_key, uname, local): uname
                for uname, local in todo
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with _write_lock:
                    completed += 1
                    print(
                        f"[{completed}/{total}] @{result['username']}  "
                        f"exists={result['account_exists']}  "
                        f"name={result['name_match']}  "
                        f"desc_sim={result['description_partial_match']}  "
                        f"loc={result['location_match']}"
                    )
                    writer.writerow(result)
                    csvfile.flush()

    print(f"\nDone. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
