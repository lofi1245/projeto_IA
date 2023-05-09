from baysian import *



# CPTable({(T, T): .99,
#             (T, F): .8,
#             (F, T): .9,
#             (F, F): .0},
#         [chuva, grama])

grama_net = (BayesNet()
             .add('chuva', [], 0.2)
             .add('regador',[],0.4)
             .add('grama', ['chuva', 'regador'], {(T, T): 0.99, (T, F): 0.8, (F, T): 0.9, (F, F): 0.0})
             .add('regador', ['chuva'], {T: 0.4, F: 0.01})
             )

# Tornar Burglary, Earthquake, etc. variáveis globais
globals().update(grama_net.lookup)