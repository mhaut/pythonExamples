# -*- encoding: utf-8 -*-

#Los datos LIDAR contienen: 
#Coordenadas X,Y y Z
#Intensidad (del retorno reflectado)
#Número de retorno
#Número de retornos 
#El numero de retorno suele ser 4 ó 5 dependiendo del equipo de escaneo.
#En el caso de los datos descargados son 4, y el escaner empleado por el IGN es ASL60 de Leica (más datos del sensor Tabla 5.5 del proyecto adjunto)


#point_write->intensity = point_read->intensity;
#point_write->return_number = point_read->return_number;
#point_write->number_of_returns_of_given_pulse = point_read->number_of_returns_of_given_pulse;
#point_write->scan_direction_flag = point_read->scan_direction_flag;
#edge_of_flight_line;
#classification;
#scan_angle_rank;
#user_data;
#point_source_ID;

import sys
import ctypes
# sudo apt-get install python-liblas
from liblas import file

f = file.File(sys.argv[1],mode='r')
outfile = open('data.csv', 'w')
outfile.write("Coordenada X, Coordenada Y, Coordenada Z, Intensidad, Numero de Retornos, Numero de Retornos, Clasificacion\n")
for p in f:
    #print "Coordenadas X,Y,Z: ", p.x, p.y, p.z
    #print "Intensidad: ", p.intensity
    #print "Número de retorno: ", p.return_number
    #print "Número de retornos del pulso: ", p.number_of_returns
    #print "Clasificación: ", p.classification
    #print "Coordenadas X,Y,Z: ", p.get_x(), p.get_y(), p.get_z()
    #print "Intensidad: ", p.get_intensity()
    #print "Número de retorno: ", p.get_return_number()
    #print "Número de retornos del pulso: ", p.get_number_of_returns()
    #print "Clasificación: ", ctypes.c_ubyte(p.get_classification())
    outfile.write(str(p.get_x()) + "," + str(p.get_y()) + "," + str(p.get_z())
                  + "," + str(p.get_intensity()) + "," + str(p.get_return_number())
                  + "," + str(p.get_number_of_returns()) + "," + str(ctypes.c_ubyte(p.get_classification()))+"\n")
outfile.close()