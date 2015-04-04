__author__="huziy"
__date__ ="$Apr 7, 2011 3:29:35 PM$"

#              KIND =0, p est en hauteur (m) par rapport au niveau de la mer (-20,000 -> 100,000)
#               KIND =1, p est en sigma                                       (0.0 -> 1.0)
#               KIND =2, p est en pression (mb)                               (0 -> 1100)
#               KIND =3, p est un code arbitraire                             (-4.8e8 -> 10e10)
#               KIND =4, p est en hauteur (M) par rapport au niveau du sol    (-20,000 -> 100,000)
#               KIND =5, p est en coordonnee hybride                          (0.0 -> 1.0)
#               KIND =6, p est en coordonnee theta                            (1 -> 200,000)
#               KIND =10, p represente le temps en heure                      (0.0 -> 200,000.0)
#               KIND =15, reserve (entiers)
#               KIND =17, p represente l'indice x de la matrice de conversion (1.0 -> 1.0e10)
#               KIND =21, p est en metres-pression  (partage avec kind=5 a cause du range exclusif)
#                                                                             (0 -> 1,000,000) fact=1e4

HEIGHT_METERS_OVER_SEA_LEVEL = 0
HEIGHT_METERS_OVER_SURFACE_LEVEL = 4
SIGMA = 1
PRESSURE = 2
ARBITRARY = 3
HYBRID = 5
THETA = 6
TIME_HOURS = 10
RESERVED = 15
METERS_AND_PRESSURE = 21
KIND_17 = 17



if __name__ == "__main__":
    print("Hello World")
