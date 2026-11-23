# Modelos desarrollados y su evaluación (actualización de experimentación en MLflow)

Esta actualización documenta la segunda iteración de experimentación del problema multietiqueta, incorporando nuevas versiones de modelos, comparándolas con la primera corrida y seleccionando la mejor alternativa con base en evidencia cuantitativa registrada en MLflow. La decisión de selección mantiene como criterio principal `validation_f1_micro` y usa `test_f1_micro`, `test_f1_macro`, precisión micro y recall micro como métricas de contraste para validar estabilidad fuera de muestra.

En la primera iteración se habían evaluado 14 corridas. En esta segunda iteración se añadieron 6 corridas nuevas (tres configuraciones adicionales probadas tanto con representación TF-IDF como con embeddings), para un total consolidado de 20 corridas versionadas. El experimento actualizado quedó registrado en MLflow con la convención de nombres de corrida por tipo de representación y configuración, lo que permite trazabilidad directa entre versión de modelo, hiperparámetros y resultados.

La siguiente tabla resume exclusivamente las seis corridas nuevas incorporadas en esta iteración.

| run_name | feature_type | model_family | selected_threshold | validation_f1_micro | test_f1_micro | test_f1_macro | test_precision_micro | test_recall_micro |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| tfidf_calibrated_linearsvc_c_0.75 | tfidf_unigram_bigram | calibrated_linear_svc | 0.30 | 0.554842 | 0.557784 | 0.425216 | 0.561989 | 0.553642 |
| tfidf_logreg_l2_c_5.0 | tfidf_unigram_bigram | logistic_regression_l2 | 0.20 | 0.549839 | 0.553232 | 0.433902 | 0.513389 | 0.599779 |
| tfidf_sgd_logloss_alpha_2e-5 | tfidf_unigram_bigram | sgd_log_loss | 0.20 | 0.541844 | 0.539256 | 0.365969 | 0.519076 | 0.561068 |
| embeddings_calibrated_linearsvc_c_0.75 | all_minilm_l6_v2_embeddings | calibrated_linear_svc | 0.25 | 0.500743 | 0.512150 | 0.368619 | 0.486091 | 0.541160 |
| embeddings_logreg_l2_c_5.0 | all_minilm_l6_v2_embeddings | logistic_regression_l2 | 0.30 | 0.498433 | 0.510066 | 0.359020 | 0.496485 | 0.524411 |
| embeddings_sgd_logloss_alpha_2e-5 | all_minilm_l6_v2_embeddings | sgd_log_loss | 0.30 | 0.492358 | 0.503495 | 0.317793 | 0.500625 | 0.506399 |

La comparación frente a la primera versión muestra que sí hubo mejora y además cambió el mejor modelo por criterio primario. En la iteración inicial, el mejor candidato era `tfidf_calibrated_linearsvc_c_1.0`; tras incorporar nuevas configuraciones, el mejor candidato pasó a ser `tfidf_calibrated_linearsvc_c_0.75`.

| Escenario | run_name | validation_f1_micro | test_f1_micro | test_f1_macro | test_precision_micro | test_recall_micro |
|:--|:--|--:|--:|--:|--:|--:|
| Mejor modelo (iteración inicial) | tfidf_calibrated_linearsvc_c_1.0 | 0.554081 | 0.553979 | 0.418398 | 0.559150 | 0.548902 |
| Mejor modelo (iteración actualizada) | tfidf_calibrated_linearsvc_c_0.75 | 0.554842 | 0.557784 | 0.425216 | 0.561989 | 0.553642 |
| Diferencia (actualizada - inicial) | - | +0.000761 | +0.003805 | +0.006818 | +0.002839 | +0.004740 |

El patrón agregado por tipo de representación también mejora con la segunda iteración. Aunque ambas familias de representación suben en promedio, TF-IDF se mantiene por encima de embeddings en las métricas centrales del objetivo.

| feature_type | validation_f1_micro (14 corridas) | validation_f1_micro (20 corridas) | Delta | test_f1_micro (14 corridas) | test_f1_micro (20 corridas) | Delta |
|:--|--:|--:|--:|--:|--:|--:|
| tfidf_unigram_bigram | 0.5235 | 0.5311 | +0.0076 | 0.5223 | 0.5306 | +0.0083 |
| all_minilm_l6_v2_embeddings | 0.4888 | 0.4913 | +0.0025 | 0.4980 | 0.5012 | +0.0032 |

En términos de decisión, la mejor alternativa para despliegue pasa a ser **`tfidf_calibrated_linearsvc_c_0.75`**, porque lidera el criterio de selección y además mejora simultáneamente las métricas de evaluación en test frente al mejor modelo anterior. El cambio de `C=1.0` a `C=0.75` sugiere que una regularización ligeramente mayor ayuda a mejorar generalización en este problema sin sacrificar precisión global.

Las nuevas versiones con embeddings también muestran avances respecto a sus pares previos, especialmente en F1 macro, pero todavía no alcanzan el nivel de desempeño de los mejores modelos TF-IDF bajo el mismo esquema de umbral global. Por tanto, embeddings se conserva como línea alternativa de investigación, mientras que TF-IDF calibrado se prioriza para operación por su mejor balance entre desempeño y costo computacional.

Como cierre de la iteración, MLflow cumple el objetivo de versionamiento al dejar trazadas las corridas por nombre, familia, hiperparámetros y métricas comparables, lo que permite justificar la selección final con evidencia reproducible y transparente.
