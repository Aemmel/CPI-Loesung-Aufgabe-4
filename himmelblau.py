# -*- coding: utf-8 -*-

def himmelblau(coordvals):
    """ Berechnung der zweidimensionalen Himmelblau-Funktion.

        Argumente
        ---------
        coordvals : float
            (2-tupel) x- und y-Koordinate (x,y)
            x und y können 1d numpy-arrays sein

        Output
        ------
        fvalues : float
            1d numpy-array mit den Funktionswerten der Himmelblau-Funktion

        Info
        ----
        Die Minima dieser Funktion mit 18 Stellen Genauigkeit:
        minima = [
              [3.00000000000000000, 2.00000000000000000],
              [-2.80511808695274485, 3.13131251825057297],
              [-3.77931025337774689, -3.28318599128616941],
              [3.58442834033049174, -1.84812652696440355]
         ];
    """

    x = coordvals[0]
    y = coordvals[1]

    fvalues = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    return fvalues
