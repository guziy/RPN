__author__ = 'huziy'


def main():
    path = "/skynet3_rech1/huziy/Global_terrain_slopes_30s/GloSlopesCl1_30as.asc"

    with open(path) as f:
        for i, line in enumerate(f):
            if i < 6:
                print(line)

            if 3000 < i < 4000:
                nums = [int(s.strip()) for s in line.split()]
                nums = [n for n in nums if n != 255]
                if len(nums):
                    print(min(nums), max(nums), len(nums))


if __name__ == "__main__":
    main()