import argparse

from analysis import analysis
from filtering import filtering
from preference_matching import preference_matching
from merge_data import merge_data

def orchestration(complete: bool, reproduction: bool) -> None:
    analysis(complete)
    filtering()
    preference_matching(reproduction)
    merge_data()

if __name__ == "__main__":

    parser = argparse.ArgumentParser("orchestration")
    parser.add_argument("-c", "--complete", help="Boolean indicating if original dataset should be reproducted",
                        default=True, action="store_false")
    parser.add_argument("-rep", "--reproduction", help="Boolean indicating if original dataset should be reproducted",
                        default=True, action="store_false")
    args = parser.parse_args()
    orchestration(complete=args.complete, reproduction=args.reproduction)