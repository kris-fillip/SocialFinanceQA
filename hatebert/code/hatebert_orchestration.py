from hatebert_classification import hatebert_classification
from hatebert_filtering import hatebert_filtering

def hatebert_orchestration():
    hatebert_classification()
    hatebert_filtering()


if __name__ == "__main__":
    hatebert_orchestration()