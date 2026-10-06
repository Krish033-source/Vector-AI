import re

MATHS_PATTERN = re.compile(r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*\??\s*$")

WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}
WORD_TO_OP = {
    "plus": "+", "add": "+", "minus": "-", "subtract": "-",
    "times": "*", "into": "*", "multiplied": "*", "multiply": "*",
    "divided": "/", "divide": "/", "by": "",
}


def _normalize_spoken_maths(text):
    words = text.strip().lower().replace("?", "").split()
    converted = []
    for w in words:
        if w in WORD_TO_NUM:
            converted.append(WORD_TO_NUM[w])
        elif w in WORD_TO_OP:
            if WORD_TO_OP[w]:
                converted.append(WORD_TO_OP[w])
        else:
            converted.append(w)
    return " ".join(converted)


def try_solve_maths(text):
    candidates = [text.strip(), _normalize_spoken_maths(text)]

    for candidate in candidates:
        match = MATHS_PATTERN.match(candidate)
        if match:
            break
    else:
        return None

    a_str, op, b_str = match.groups()
    a, b = int(a_str), int(b_str)

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            return "undefined"
        result = a / b
        if result == int(result):
            result = int(result)
        else:
            result = round(result, 2)

    return str(result)


if __name__ == "__main__":
    tests = ["2 + 3", "12 - 4", "7*8", "100 / 4", "9 / 2", "not maths", "5 +", "10-3"]
    for t in tests:
        print(f"{t!r:15s} -> {try_solve_maths(t)}")
