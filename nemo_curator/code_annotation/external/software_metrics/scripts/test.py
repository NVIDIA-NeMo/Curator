import software_metrics

LANGUAGE = "Python"
CODE = "print(2)\nx=3\n#comment\n\n\n\n\n"

if __name__ == "__main__":
    metrics = software_metrics.get_all_metrics(LANGUAGE, CODE)
    print(metrics)
