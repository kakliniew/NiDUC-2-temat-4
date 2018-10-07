from typing import Any, Union
 
import numpy as np
from scipy import misc as mc
import matplotlib.pyplot as plt
import random
from random import randrange, uniform
 
nowy = np.array([], dtype=int)
img2 = mc.imread('123123.png', True, 'L')
img2_x = img2.shape[1]
img2_y = img2.shape[0]
blad=0
 
def macierz(A):
    m, n = A.shape
    B = np.empty([n, m], dtype=int)
 
    for i in range(0, n):
        for j in range(0, m):
            B[i][j] = A[j][i]
    return B
 
# funkcja z wiki obudowana tak zeby przyjmowała zamiast input_bitsring
# jakis array intów. Wielomin i initial_filler dalej podawane są jko string
def crc(wejscie, wielomian, costam_co_nie_wiem_czym_jest):
    def crc_remainder(input_bitstring, polynomial_bitstring, initial_filler):
        '''
    Calculates the CRC remainder of a string of bits using a chosen polynomial.
    initial_filler should be '1' or '0'.
    '''
        len_input = len(input_bitstring)
        initial_padding = initial_filler * (len(polynomial_bitstring) - 1)
        input_padded_array = list(input_bitstring + initial_padding)
        polynomial_bitstring = polynomial_bitstring.lstrip('0')
        while '1' in input_padded_array[:len_input]:
            cur_shift = input_padded_array.index('1')
            for i in range(len(polynomial_bitstring)):
                if polynomial_bitstring[i] == input_padded_array[cur_shift + i]:
                    input_padded_array[cur_shift + i] = '0'
                else:
                    input_padded_array[cur_shift + i] = '1'
        return ''.join(input_padded_array)[len_input:]
 
    # wejscie -> costam = np.array([10101])
    wynik2 = np.array([], dtype=int)
 
    tmp = np.array_str(wejscie)
    tmp = tmp.strip('[')
    tmp = tmp.strip(']')
    tmp = tmp.replace(' ', '')
    wynik = crc_remainder(tmp, wielomian, costam_co_nie_wiem_czym_jest)
    for a in wynik:
        a = int(a)
        wynik2 = np.append(wynik2, a)
    return wynik2
 
 
 
def Szum(dane):                     # sztuczny szum w przesyle, na razie szansa na zmiane 1% , mozna edytowac zaleznie od wynikow
    x = random.randint(1,10000)
    if(x <= 3125):
        if(dane == 0):
            dane = 1
        else:
            dane = 0
    return dane
 
def Gilbertcondition(stan):           #czy zmieniac stan
    if (stan == 0):                  # stan dobry
        losowa1 = random.randint(1,10)
        if(losowa1 <= 3):           #prawdopodobienstwo na zmiane stanu 0,3 jesli dobry
            stan = 1
            return stan  # zwraca wartosc dla gilberta w odpowiednim stanie
        else:
            return stan
    else:
        losowa2 = random.randint(1,10)
        if(losowa2 <= 5):               # prawdopodoienstwo zmiany stanu 1/2
            stan = 0
            return stan
        else:
            return stan
 
 
def Gilbertsend(x,stan):
    if(stan == 0):                # dobry stan (dwa stany dobry = 0, zły = 1)
        z = random.randint(1,10000)
        if (z == 5):            # jesli jest w dobrym stanie i wylosuje, ze akurat robi blad to zamiana(szansa 0,0001)
            if (x == 0):
                x=1
                return x
            if (x == 1):
                x=0
                return x
        else:
            return x           # jesli dobry stan i wszystko ok to wysyla x
    else:                       # else - czyli mamy zly stan (czyli wartosc 1)
        y: int = random.randint(0,1)    # losuje liczbe 0 lub 1
        if(y == 1):             # jesli 1 to robi blad, tzn zamienia wartosc na przeciwna
            if(x == 0):
                x=1
                return x
            if(x == 1):
                x=0
                return x
        if(y == 0):
            return x
 
 
 
def HammingCoding(x):
    G = ([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,1,1,1],[1,0,1,1],[1,1,0,1])
    tmp = np.matmul(G,x)
    licznik = 0
    while licznik < 7:
        tmp[licznik][0]=tmp[licznik][0]%2
        licznik += 1
    return tmp
 
 
 
def HammingEncoding(x):
    H = [[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]]
    tmp = np.matmul(H, x)
    licznik = 0
    while licznik < 3:
        tmp[licznik] = tmp[licznik] % 2
        licznik += 1
    return tmp
 
 
def ParityBitAdd(x):
    a = np.count_nonzero(x)
    a = a % 2
    x = np.append(x, a)
    return x
 
 
def ParityBitCheck(x):
    a = np.count_nonzero(x)
    a = a % 2
    if a == 0:
        return True
    elif a == 1:
        return False
 
 
def TMRCoding(x):
    x = np.vstack((x, x, x, x, x))
 
    return x
 
 
def TMREncoding(x, cos):
    counter = 0
    result = np.empty([0,0], dtype=int)
    for t in range(0,cos):
        tmp = (x[0+counter] and x[1+counter] and x[2+counter]) or (x[0+counter] and x[1+counter] and x[3+counter]) or (x[0+counter] and x[1+counter] and x[4+counter]) or (x[1+counter] and x[2+counter] and x[3+counter]) or (x[1+counter] and x[2+counter] and x[4+counter]) or (x[2+counter] and x[3+counter] and x[4+counter]) or (x[0+counter] and x[2+counter] and x[3+counter]) or (x[0+counter] and x[2+counter] and x[4+counter]) or (x[0+counter] and x[4+counter] and x[3+counter]) or (x[1+counter] and x[3+counter] and x[4+counter])
        result = np.append(result, tmp)
        counter += 5
    return result
 
def TMREncoding2(x, cos):
 
    counter = 0
    result = np.empty([0, 0], dtype=int)
    for t in range(0, cos):
        tmp = x[counter]+x[counter+cos]+x[counter+cos+cos]+x[counter+cos+cos+cos]+x[counter+cos+cos+cos+cos]
        if tmp>2:
            tmp = 1
        else:
            tmp = 0
        result = np.append(result, tmp)
        counter += 1
    return result
 
 
 
 
 
stan = 1
 
 
 
 
 
 
 
 
 
### moja czesc
 
###########zmienia obrazek na ciąg liczb dwójkowych
for x in img2:
    for liczba in x:
        ile = 8
        while ile > 0:
            znak = liczba - (2 ** (ile - 1))
            if znak >= 0:
                nowy = np.append(nowy, [1])
                liczba = liczba - (2 ** (ile - 1))
            else:
                nowy = np.append(nowy, [0])
            ile = ile - 1
 
nowy7 = np.empty([], dtype=int)
nowy3 = np.array([1,4], dtype=int)
nowy4= np.array([1,4], dtype=int)
nowy2= np.array([4,1], dtype=int)
nowy5= np.array([4,1], dtype=int)
ile = 0  # licznik do pętli ponizej
obieg = 0  # też licznik do pętli, potrzebny do wybierania kolejnych bitow z ciagu
for x in nowy:
    ile = ile + 1
ile = ile//8
 
ile2 = ile
print(ile)
print(nowy7.size)
while ile>0:
 
            nowy3 = nowy[(0+obieg*8):(4+obieg*8)]
            nowy4 = nowy[(4+obieg*8):(8+obieg*8)]
 
            nowy5 = np.reshape(nowy3,(4,1))
            nowy2 = np.reshape(nowy4, (4, 1))
 
 
            #z=np.asmatrix(nowy5)
            #y=np.asmatrix(nowy2)
            #print(z)
            #print(y)
            wynik = HammingCoding(nowy5)
            wynik2 = HammingCoding(nowy2)
 
 
            wynik3 = np.append(wynik,wynik2)
 
            doprzeslania = np.append(wynik3, crc(wynik3, '1011', '0'))
            licznik1 = 0
            odebrane = np.empty([0, 0], dtype=int)
            while licznik1 < 17:
                stan = Gilbertcondition(stan)
                odebrane = np.append(odebrane, Gilbertsend(doprzeslania[licznik1], stan))
 
                licznik1 += 1
 
            prawdaa = True
 
 
 
            iterator = 0
            for x1 in crc(odebrane[0:13], '1011', '0'):
                if (x1 != odebrane[13 + iterator]):
                    prawdaa = False
                iterator += 1
 
            nr_miejsca1 = 0
            nr_miejsca2 = 0
 
            # po tym nie dziala
            if (prawdaa):
                print("Przesłano poprawnie")
 
            else:
                potega_dwa1 = 4  ## blad pierwszy hamming
 
                for kl in HammingEncoding(odebrane[0:7]):
                    nr_miejsca1 = nr_miejsca1 + kl * potega_dwa1
 
                    potega_dwa1 = potega_dwa1 / 2
 
                    ## blad drugi hamming
 
                potega_dwa2 = 4  ## blad pierwszy hamming
 
                for kl in HammingEncoding(odebrane[7:14]):
                    nr_miejsca2 = nr_miejsca2 + kl * potega_dwa2
 
                    potega_dwa2 = potega_dwa2 / 2
 
            print(nr_miejsca1)
            print(nr_miejsca2)
            if (int(nr_miejsca1) != 0):
                odebrane[int(nr_miejsca1) - 1] = (odebrane[int(nr_miejsca1) - 1] + 1) % 2
 
            wynik_przeslany1 = np.append(odebrane[0:3], odebrane[4])
 
            if (int(nr_miejsca2) != 0):
                odebrane[int(nr_miejsca2) - 1 + 7] = (odebrane[int(nr_miejsca2) - 1 + 7] + 1) % 2
            wynik_przeslany2 = np.append(odebrane[7:10], odebrane[11])
 
            przeslana_calosc = np.append(wynik_przeslany1, wynik_przeslany2)
            kl=0
            test= True
 
            while kl<4:
                if(nowy3[kl]!=wynik_przeslany1[kl] or nowy4[kl] != wynik_przeslany2[kl]):
                    test = False
 
                kl=kl+1
            if (test == True):
                blad = blad + 1
 
 
 
            print("przeslane wartosci po naprawieniu to:")
            print(przeslana_calosc)
            # za duzo wkleilem
 
 
 
 
 
            if ile==ile2:
                nowy7=przeslana_calosc
            else:
                nowy7 = np.append(nowy7,przeslana_calosc)
            ile-=1
            obieg+=1
 
 
 
 
 
 
    # wysylamy-odbieramy-jeslinak powtorz/jestliak obieg+3 i od nowa
print(obieg)
print(nowy7)
print(nowy7.size)
print(nowy.size)
# nowy7 = np.append(nowy7,[0,0])
# składanie na nowow w obrazek
licznik10 = 0
obrazek_gotowy = np.array([], dtype=float)
obrazek_wiersz = np.array([], dtype=float)
 
for a in range(0, img2_x * img2_y):
    liczba10 = 0
    licznik11 = 7
    for b in range(0, 8):
        liczba10 += (2 ** licznik11) * nowy7[licznik10]
        licznik11 -= 1
        licznik10 += 1
    obrazek_wiersz = np.append(obrazek_wiersz, liczba10)
obrazek_gotowy = np.reshape(obrazek_wiersz, (img2_y, img2_x))
 
 
f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(obrazek_gotowy, cmap=plt.cm.gray)
f.add_subplot(1, 2, 2)
plt.imshow(img2, cmap=plt.cm.gray)
plt.show()
print(blad)
