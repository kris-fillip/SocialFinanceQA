from config.constants import RICHTEXT_ATTRIBUTES


def get_basic_attribute(obj: dict, attribute: str) -> None | int | float | str:
    if attribute not in obj or obj[attribute] is None or obj[attribute] == "":
        return None
    else:
        return obj[attribute]


def get_richtext(obj: dict, attribute: str) -> None | str:
    if attribute not in obj:
        return None
    else:
        link_flair = obj[attribute]
        if len(link_flair) == 0:
            return None
        if not get_basic_attribute(link_flair[0], "t"):
            if not get_basic_attribute(link_flair[0], "a"):
                return None
            return link_flair[0]["a"]
        return link_flair[0]["t"]


def get_attribute(obj: dict, attribute: str) -> None | int | float | str:
    if attribute in RICHTEXT_ATTRIBUTES:
        return get_richtext(obj, attribute)
    else:
        return get_basic_attribute(obj, attribute)
