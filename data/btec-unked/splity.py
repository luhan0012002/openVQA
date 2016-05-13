f = open('./test.zh-en', 'r')
f_src = open('./test.src', 'w')
f_tgt = open('./test.tgt', 'w')
lines = f.readlines()
for line in lines:
    tmp = line.split('|||')
    src = tmp[0] + '\n'
    tgt = tmp[1] 
    f_src.write(src)
    f_tgt.write(tgt)
f.close()
f_src.close()
f_tgt.close()

