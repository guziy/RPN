__author__ = 'huziy'

import matplotlib.pyplot as plt
from pathlib import Path


def test_save(im_format="png", ask_input=False):
    img_path = "test.{}".format(im_format)
    fig = plt.figure()
    data = [i for i in range(20)]
    plt.plot(data)
    fig.tight_layout()

    try:
        fig.savefig(img_path)
        plt.close(fig)
        resp = ""

        if ask_input:
            resp = input("Do you want to check the generated image? (yes/no):").strip().lower()

        if resp != "yes":
            p = Path(img_path)
            p.unlink()

    except Exception as exc:
        print("Error: Saving {} does not work")
        print(exc)
        return -1

    return 0


def test_save_all_formats():
    formats_to_test = ["png", "pdf", ]
    test_results = []
    for im_format in formats_to_test:
        st = test_save(im_format=im_format)
        msg = "ok"
        if st < 0:
            msg = "failed"
        test_results.append("{}: {}".format(im_format, msg))

    print(test_results)




if __name__ == '__main__':
    for _ in range(10):
        test_save_all_formats()