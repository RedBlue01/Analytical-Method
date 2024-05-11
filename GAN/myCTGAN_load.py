from sdv.single_table import CTGANSynthesizer
synthesizer=CTGANSynthesizer.load(
    filepath='/home/visitor/Huang/Analytical-Method/GAN/CTGANsynthesizer_e100.pkl'
)
# 设置已拟合
# print('!!!'+str(synthesizer._fitted))
# synthesizer.fitted=True
# synthesizer = CTGANSynthesizer(
#     metadata, # required
#     enforce_rounding=True,
#     epochs=20,
#     verbose=True
# )
print(synthesizer.get_parameters())

# synthesizer.fit(data)
# print(synthesizer.get_loss_values())

synthetic_data = synthesizer.sample(num_rows=500)

print(synthetic_data)
print('Done')