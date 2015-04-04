"""
module pour traiter des fichiers ccc
"""

import struct
import numpy as N
import time
import sys




class champ_ccc :
    """
    champ_ccc
    un champ_ccc est une liste de dictionnaires composes des clefs
    ibuf1, ibuf2, ... jusqu'a ibuf8 et d'un tableau x(ibuf5 x ibuf6)
    """
    def __init__(self,fichier='') :
        # mettre des options selon qu'on fourni un nom de fichier ou non
        self.fichier=fichier # nom du fichier ccc
        if self.fichier != '' :
            self._u=open(self.fichier)


    def _ferm_fichier(self) :
        """
        methode qui ferme l'unite _u
        """
        close(self._u)

    def _rewind_fichier(self) :
        """
        methode qui rewind l'unite _u
        """
        self._u.seek(0)

    class _message_fin_fichier : pass

    def _lit_ibuf(self) :
        """
        methode pour lire une etiquette de champ
        sortie = ibuf, un dictionnaire contenant les clef ibuf1, ibuf2, ..ibuf8
        """
        # on passe par dessus l'etiquette du debut de record
        self._u.seek(4,1)

        # lecture du ibuf
        ibuf={}
        #lecture des ibuf1 a 8
        format='>4sxxxxq4sxxxx5q'
        long_rec = 64

        # si la string lue est vide, le fichier est tout lu
        string_lue = self._u.read(long_rec)
        if string_lue == '' :
            # on rewind le fichier et on envoi une exception
            self._u.seek(0)
            raise self._message_fin_fichier()
        
        # on decode le ibuf
        ibuf_lu = struct.unpack(format,string_lue)
        # on attribue les etiquettes lues au ibuf
        for i in range(8) :
            clef = 'ibuf'+str(i+1)
            val = ibuf_lu[i]
            ibuf[clef] = val

        # on avance par dessus l'etiquette de fin du record
        self._u.seek(4,1)

        # on retourne le ibuf sous forme de dictionnaire
        return ibuf

    def _decode_champ(self,ibuf,lecture=0,debug=0,option_lecture=1) :
        """
        methode qui decode un champ selon le packer
        arguments entree :

        ibuf : le ibuf correspondant au champ
        lecture : poser lecture=0 pour ne pas lire le champ (defaut)
                               =1 pour que le champ decode soit fourni en sortie

        ex
        champ=x._decode_champ(ibuf,lecture=1)
        champ contient le champ lu et decode
        champ=x._decode_champ(ibuf,lecture=0) ou  champ=x._decode_champ(ibuf)       
        champ n contient rien

        note : afin de tenter d'acclerer le traitement, une clef option de lecture a ete
        ajoutee. Sa valeur peut etre de 1 (defaut) ou 2.
        si option_lecture=1 : la lecture se fait avec un objet struct qui est ensuite depacte
           option_lecture=2 : la lecture se fait avec un objet array

        des tests ont montre que tel quel, le code est un peu plus
        rapide avec option_lecture=1 mais que les resultats ne sont
        pas sensibles.

        """
        import struct

        # lecture de l'etiquette du debut du record
        # long_rec est ecrit en 32 bit (i.e. 4 bytes)
        long_rec = struct.unpack('>l',self._u.read(4))[0]

        if debug : print('longueur du record=',long_rec)

        if lecture : 
        
            # si lecture = 1 on lit le champ

            # ajustement selon le packing
            npack=ibuf['ibuf8']

            # on lit xmin,xmax si necessaire 
            if npack != 1 :
                # xmin et xmax sont ecrits en 64 bits (2 chiffres de 8 bytes chaque)
                xmin,xmax=struct.unpack('>2d',self._u.read(16))
                # on ajuste la longueur du record a lire
                long_rec=long_rec-16

                if debug : print('xmin,xmax=',xmin,xmax)
                if debug : print('long_rec apres lecture xmin,xmax=',long_rec)
            

            # choix du format selon le packing
            #format={1:'d',2:'l',4:'H'}[npack]
            # nouveau format car probleme sous python2.6 avec packing a 1 et 2
            format={1:'d',2:'I',4:'H'}[npack]
            if debug : print('format=',format)

            # calcul de xscali (voir paccrn)
            if npack != 1 :
                if npack == 2 :
                    biggest=2**31-1
                else :
                    nbits = 64/npack
                    biggest = 2**nbits-1
                ecart = xmax-xmin
                xscali = ecart/biggest

            # lecture du champs en temps que tel
            #
            # on calcule la longueur minimum a lire
            nij=ibuf['ibuf5']*ibuf['ibuf6']
            if option_lecture == 1 :
                if debug : print('lecture option 1')
                itemsize=struct.calcsize(format)
                nb_elements_a_lire=long_rec/itemsize
                format='>'+str(nb_elements_a_lire)+format
                if debug :
                    print('format=',format)
                    print('itemsize=',itemsize)
                    print('long_rec=',long_rec)
                    print('nij*itemsize=',nij*itemsize)
                vec_lu=struct.unpack(format,self._u.read(long_rec))
                if debug : sys.exit()
            elif option_lecture == 2 :
                if debug : print('lecture option 2')
                import array
                vec_lu=array.array(format)
                vec_lu.fromstring(self._u.read(long_rec))
                itemsize=vec_lu.itemsize
                # si machine avec little indian, on
                # ajuste le champ lu
                if sys.byteorder == 'little' :
                    if debug : print('correction little endian')
                    vec_lu.byteswap()
                
            # on ajuste la longueur du vecteur lu selon
            # les blancs lus
            nb_elements_lus = long_rec/itemsize
            if debug : print('itemsize=',itemsize)
            if debug : print('nb_elements_lus=',nb_elements_lus)
            if nb_elements_lus != nij :
                # on enleve les elements lus pour rien
                nb_val_par_64bits=8/itemsize
                ###nb_val_par_64bits=8/vec_lu.itemsize
                nb_elements_a_enlever=nb_val_par_64bits-nij%nb_val_par_64bits
                if nb_elements_a_enlever > 0 :
                    if debug : print('on enleve ',nb_elements_a_enlever,' elements a vec_lu')
                    if debug : print('vec_lu avant =',vec_lu)
                    vec_lu=vec_lu[:-nb_elements_a_enlever]
                    if debug : print('vec_lu apres =',vec_lu)
                

            ni=ibuf['ibuf5']
            nj=ibuf['ibuf6']

            # verification de la coherence de taille des champs lus
            if ni*nj != len(vec_lu) :
                print(50*'*')
                print('PROBLEME')
                print("champ lu n'a pas la meme grosseur que specifie dans ibuf")

            # option
            # on tente differentes option pour accelerer le code
            option=1

            if option == 1 :
                champ=N.zeros((ni,nj),'Float64')
                for j in range(nj) :
                    for i in range(ni) :
                        indice=i+j*ni
                        champ[i,j]=vec_lu[indice]

            elif option == 2 :
                champ=N.array(vec_lu,shape=(nj,ni),type='Float64')
                champ=N.transpose(champ)

            # decodage si necessaire
            if npack !=1 : champ=champ*xscali+xmin

            if debug == 1 :
                for i in range(1,ni+1) :
                    for j in range (1,nj+1) :
                        print('(%5i,%5i) val=%14.6e' % (i,j,champ[i-1,j-1]))

        else :
            # on ne fait qu'avancer le fichier
            self._u.seek(long_rec,1)

        # on passe par dessus l'etiquette de fin du record
        self._u.seek(4,1)
        if lecture : return champ

    def invntry(self) :
        """
        methode qui affiche les etiquettes des champs dans le fichier
        """
        # boucle sur le fichier
        compteur = 0
        entete="""
        invntry de %s
        #record   ibuf
        """
        # on fait revenir le fichier au debut
        self._rewind_fichier()

        while 1 :
            # lecture du ibuf jusqu'a la fin du fichier
            try :
                ibuf=self._lit_ibuf()
            except self._message_fin_fichier :
                print('lecture de ',self.fichier,' terminee')
                break
            compteur+=1
            if compteur == 1 : print(entete % self.fichier)
            print("%5i %4s %10i %4s %5i %5i %5i %5i %5i" % (compteur,ibuf['ibuf1'],ibuf['ibuf2'],ibuf['ibuf3'],ibuf['ibuf4'],ibuf['ibuf5'],ibuf['ibuf6'],ibuf['ibuf7'],ibuf['ibuf8']))
            self._decode_champ(ibuf)
                                 

    def ggstat(self,debug = 0,option_lecture = 1) :
        """
        methode qui fait un ggstat des records d'un fichier ccc
        """

        compteur=0
        entete="""
        ggstat de %s
        #record   ibuf
        """
        # on fait revenir le fichier au debut
        self._rewind_fichier()

        while 1 :
            # lecture du ibuf jusqu'a la fin du fichier
            try :
                ibuf=self._lit_ibuf()
            except self._message_fin_fichier :
                print('lecture de ',self.fichier,' terminee')
                break
            compteur+=1
            champ=self._decode_champ(ibuf,lecture=1,debug=debug,option_lecture=option_lecture)
            if compteur == 1 : print(entete % self.fichier)
            var=(champ-champ.mean())**2
            var=var.mean()
            print("%5i %4s %10i %4s %5i %5i %5i %5i %5i %14.6e %14.6e %14.6e %14.6e" % (compteur,ibuf['ibuf1'],ibuf['ibuf2'],ibuf['ibuf3'],ibuf['ibuf4'],ibuf['ibuf5'],ibuf['ibuf6'],ibuf['ibuf7'],ibuf['ibuf8'],champ.min(),champ.max(),champ.mean(),var))
            
            
    def charge_champs(self, debug = 0, i1 = None,i2 = None, j1 = None, j2 = None):
        """
        methode qui charge un champ ccc en memoire
        la sortie est une liste de [ibuf,champ]
        """
        compteur = 0
        entete = "chargment de %s # ibuf"
        t1 = time.time()
        champ_lu = [ ]
        while 1: 
            # lecture du ibuf jusqu'a la fin du fichier
            try :
                ibuf = self._lit_ibuf()
            except self._message_fin_fichier :
                if debug:
                    t2 = time.time()
                    dt = t2-t1
                    print('lecture de ',self.fichier,' terminee')
                    #print compteur,' champ lus'
                    txt = ('%10i champs lus en %3im%2is') % (compteur,dt/60,dt%60)
                    print(txt)
                return champ_lu


            compteur += 1
            champ=self._decode_champ(ibuf, lecture = 1, debug = debug)
            # extraction de la region [i1:i2,j1:j2] s'il y a lieu
            if i1 != None and i2 != None and j1 != None and j2 != None :
                #
                # on extrait la region et on modifie le ibuf en consequence
                champ = champ[i1-1:i2-1,j1-1:j2-1].copy()
                ni=i2-i1
                nj=j2-j1
                ibuf['ibuf5'] = ni
                ibuf['ibuf6'] = nj
                #
            if compteur == 1 and debug : print(entete % self.fichier)
            if debug:
                print("%5i %4s %10i %4s %5i %5i %5i %5i %5i" % (compteur,ibuf['ibuf1'],ibuf['ibuf2'],ibuf['ibuf3'],ibuf['ibuf4'],ibuf['ibuf5'],ibuf['ibuf6'],ibuf['ibuf7'],ibuf['ibuf8']))
            record={'ibuf':ibuf,'field':champ}
            champ_lu.append(record)
        

##################################################
# partie test
##################################################
import application_properties
application_properties.set_current_directory()
if __name__ == '__main__' :
    print('appel en programme')

    option_test=1


    test_path = 'RUNOFF/AET/TOTAL/aet_p1rof_196101.ccc'

    import matplotlib.pyplot as plt
    fichier = test_path
    print('file: ',test_path)
    x = champ_ccc(fichier = test_path)
    x.ggstat(debug = 0, option_lecture = 2)
    field = x.charge_champs(0)
    plt.imshow(field[0]['field'])
    plt.savefig('field.png')
    print(field[0]['ibuf'])


