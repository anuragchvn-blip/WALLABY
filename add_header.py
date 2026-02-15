# Add header to cosmicflows4.csv
header = "PGC,1PGC,T17,Vcmb,DM,e_DM,DMsnIa,e_DMsnIa,DMtf,e_DMtf,DMfp,e_DMfp,DMsbf,e_DMsbf,DMsnII,e_DMsnII,DMtrgb,e_DMtrgb,DMceph,e_DMceph,DMmas,e_DMmas,RAJ2000,DEJ2000,GLON,GLAT,SGL,SGB,CF3,Sloan,Simbad,NED,LEDA"

with open('e:/WALLABY/data/cosmicflows4.csv', 'r') as f:
    content = f.read()

with open('e:/WALLABY/data/cosmicflows4.csv', 'w') as f:
    f.write(header + '\n')
    f.write(content)

print("Header added to cosmicflows4.csv")
