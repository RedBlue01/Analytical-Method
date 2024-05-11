import pandas as pd
real_data = pd.read_csv('/home/visitor/Huang/Analytical-Method/GAN/column_123mini.csv')
synthetic_data=pd.read_csv('/home/visitor/Huang/Analytical-Method/GAN/CTGAN_sample.csv')
# 创建元数据
from sdv.metadata import SingleTableMetadata
metadata=SingleTableMetadata()
metadata.detect_from_dataframe(real_data)
print(metadata)

from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot

# 1. perform basic validity checks
diagnostic = run_diagnostic(real_data, synthetic_data, metadata, verbose=True)

# 2. measure the statistical similarity
quality_report = evaluate_quality(real_data, synthetic_data, metadata)

# 3. plot the data
fig = get_column_plot(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata,
    column_name='电流'
)

fig.write_image('/home/visitor/Huang/Analytical-Method/GAN/evaluate电流.png')