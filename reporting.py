"""
Sonuç Raporlama
"""

from config import OUTPUTS_DIR


def print_result_summary(best_model_name, validation_df, test_metrics, best_params):
    """Kapsamli sonuc yorumu yazdirir."""
    print("\n" + "=" * 70)
    print("18. SONUC YORUMU")
    print("=" * 70)

    matching_rows = validation_df.loc[validation_df["Model"] == best_model_name]
    if matching_rows.empty:
        raise ValueError(f"Validation tablosunda secilen model bulunamadi: {best_model_name}")
    best_val = matching_rows.iloc[0]

    print(f"""
OZET DEGERLENDIRME
------------------
En iyi model: {best_model_name}

Validation Sonuclari:
  Accuracy:  {best_val['Accuracy']:.4f}
  Precision: {best_val['Precision']:.4f}
  Recall:    {best_val['Recall']:.4f}
  F1-Score:  {best_val['F1-Score']:.4f}
  ROC-AUC:   {best_val['ROC-AUC']:.4f}

Test Sonuclari (pozitif sinif = malignant):
  Accuracy:  {test_metrics['accuracy']:.4f}
  Precision: {test_metrics['precision']:.4f}
  Recall:    {test_metrics['recall']:.4f}
  F1-Score:  {test_metrics['f1_score']:.4f}
  ROC-AUC:   {test_metrics['roc_auc']:.4f}

En Iyi Hiperparametreler:
""")
    if best_params:
        for k, v in best_params.items():
            print(f"  {k} = {v}")
    else:
        print("  (varsayilan parametreler)")

    print(f"""
YORUM
-----
1. Model Secimi: {best_model_name}, validation setinde en yuksek malignant F1
   skorunu elde ettigi icin; esitlikte recall, sonra da acik model onceligi
   kullanilarak secilmistir.

2. Malignant Recall: Gercek kotu huylu tumorlerin ne kadarini dogru tespit
   ettigimizi gosterir. Tibbi uygulamalarda kritiktir.

3. False Negative Onemi: Malignant tumoru benign olarak yanlis siniflandirmak
   klinik acidan cok riskli olup tedavide gecikmelere yol acabilir.

4. Onemli Oznitelikler: Tumor boyutu (area, radius, perimeter), sekil
   duzensizligi (compactness, concavity) ve doku yapisi (texture) ile iliskilidir.

5. Sinirliliklar:
   - Veri seti yalnizca 569 ornek icerir; genelleme performansi farkli olabilir.
   - Bu model yalnizca egitim amaclidir; gercek tibbi teshis araci olarak KULLANILAMAZ.
""")

    print("=" * 70)
    print("Calisma tamamlandi.")
    print(f"Cikti dosyalari: {OUTPUTS_DIR}")
    print("=" * 70)
