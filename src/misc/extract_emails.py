

def main():
    path = "/Users/huziy/Desktop/happy_holidays_emails_2017.txt"


    fout = open("/Users/huziy/Desktop/happy_holidays_emails_2017_clean.txt", "w")

    with open(path) as fin:
        for line in fin:
            line = line.strip().replace(",", "")
            email = [tok for tok in line.split() if "@" in tok][0]
            if email[0] == "<" and email[-1] == ">":
                email = email[1:-1]

            fout.write(email + "\n")

    fout.close()





if __name__ == '__main__':
    main()