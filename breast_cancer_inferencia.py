from pickle import load

#Nova paciente ja no formato do get_dummies (mesma ordem de colunas do treinamento)
nova_paciente=[[
    3,                                  #deg-malig
    0, 0, 1, 0, 0, 0,                   #age: 20-29, 30-39, 40-49, 50-59, 60-69, 70-79
    0, 0, 1,                            #menopause: ge40, lt40, premeno
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,    #tumor-size: 0-4, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 5-9, 50-54
    0, 0, 0, 0, 1, 0, 0,                #inv-nodes: 0-2, 12-14, 15-17, 24-26, 3-5, 6-8, 9-11
    0, 0, 1,                            #node-caps: ?, no, yes
    1, 0,                               #breast: left, right
    0, 0, 0, 1, 0, 0,                   #breast-quad: ?, central, left_low, left_up, right_low, right_up
    0, 1                                #irradiat: no, yes
]]
normalizador = load(open('breast_cancer_normalizador.model', 'rb'))
nova_paciente=normalizador.transform(nova_paciente)

#Carregar o modelo de melhor acuracia (Random Forest)
cancer_modelo = load(open('breast_cancer_rf.model', 'rb'))

diagnostico_nova_paciente = cancer_modelo.predict(nova_paciente)
print(diagnostico_nova_paciente)
