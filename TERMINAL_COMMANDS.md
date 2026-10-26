# دليل أوامر التنفيذ في الطرفية (Terminal CLI Guide)
### Crypto Price Prediction AI — أوامر التنبؤ بالدقائق والساعات

يوفر هذا الدليل جميع الأوامر والخيارات المتاحة لتشغيل نظام التنبؤ الذكي عبر موجه الأوامر (**PowerShell**, **Command Prompt CMD**, أو **Bash/Terminal**).

---

## 1. طرق التشغيل المتاحة

يمكنك التشغيل بثلاث طرق:
1. **باستخدام ملف الـ Batch السريع (`run_predict.bat`)** *(الأسهل في ويندوز)*
2. **باستخدام سكريبت PowerShell (`.\run_predict.ps1`)**
3. **باستخدام بايثون مباشرة عبر بيئة المشروع الافتراضية**

---

## 2. المعاملات والخيارات (Arguments & Parameters)

| المعامل | الوصف | القيم المدعومة | القيمة الافتراضية |
| :--- | :--- | :--- | :--- |
| `--symbol` | زوج العملات الرقمية | `BTCUSDT`, `ETHUSDT`, `SOLUSDT` ... | `BTCUSDT` |
| `--interval` | الفاصل الزمني للشموع | الدقائق: `1m`, `3m`, `5m`, `15m`, `30m`<br>الساعات: `1h`, `2h`, `4h`<br>اليومي: `1d` | `1h` |
| `--horizon` | أفق التنبؤ (عدد الشموع للأمام) | أي رقم صحيح (مثل `1`, `3`, `5`, `12`...) | `1` |
| `--no-charts` | تعطيل توليد المخططات (لتسريع التنفيذ) | لا يحتاج قيمة (Flag) | غير مفعل (يتم توليد الشارت تلقائياً) |

> **ملاحظة:** لمعرفة المدة الإجمالية للتنبؤ: اضرب الفاصل الزمني `--interval` في الأفق `--horizon`.  
> مثال: فاصل `5m` مع أفق `3` = تنبؤ بعد **15 دقيقة** للأمام.

---

## 3. أمثلة التنبؤ بالدقائق (Scalping & Intraday)

### تنبؤ بعد دقيقة واحدة فقط (1 Minute Ahead):
```powershell
.\run_predict.bat --symbol BTCUSDT --interval 1m --horizon 1
```

### تنبؤ بعد 5 دقائق:
```powershell
.\run_predict.bat --symbol BTCUSDT --interval 5m --horizon 1
```

### تنبؤ بعد 15 دقيقة (3 شمعات من فاصل 5 دقائق):
```powershell
.\run_predict.bat --symbol BTCUSDT --interval 5m --horizon 3
```

### تنبؤ بعد نصف ساعة لعملة إيثيريوم (ETHUSDT):
```powershell
.\run_predict.bat --symbol ETHUSDT --interval 15m --horizon 2
```

### تنبؤ بعد 45 دقيقة لعملة سولانا (SOLUSDT):
```powershell
.\run_predict.bat --symbol SOLUSDT --interval 15m --horizon 3
```

---

## 4. أمثلة التنبؤ بالساعات (Swing & Multi-Hour)

### تنبؤ بعد ساعة واحدة (1 Hour Ahead - الافتراضي):
```powershell
.\run_predict.bat --symbol BTCUSDT --interval 1h --horizon 1
```

### تنبؤ بعد 4 ساعات:
```powershell
.\run_predict.bat --symbol BTCUSDT --interval 1h --horizon 4
```
*أو باستخدام فاصل الـ 4 ساعات مباشرة بشمعة واحدة:*
```powershell
.\run_predict.bat --symbol BTCUSDT --interval 4h --horizon 1
```

### تنبؤ بعد 12 ساعة لعملة ETH:
```powershell
.\run_predict.bat --symbol ETHUSDT --interval 1h --horizon 12
```

### تنبؤ بعد 24 ساعة (يوم كامل للأمام):
```powershell
.\run_predict.bat --symbol BTCUSDT --interval 1h --horizon 24
```

---

## 5. التشغيل السريع بدون رسم بياني (Fast Terminal Mode)

إذا كنت ترغب في الحصول على الأرقام والاتجاه وإعدادات الصفقة فقط في الطرفية بأقصى سرعة دون توليد ملفات HTML و PNG:

```powershell
.\run_predict.bat --symbol BTCUSDT --interval 5m --horizon 3 --no-charts
```

---

## 6. التشغيل عبر PowerShell مباشرة

```powershell
.\run_predict.ps1 -symbol BTCUSDT -interval 5m -horizon 3
```

---

## 7. التشغيل باستخدام Python مباشرة من بيئة العمل

إذا كنت داخل المجلد الرئيسي للمشروع:

```powershell
& "..\..\..\envairoment\.venv\Scripts\python.exe" predict.py --symbol BTCUSDT --interval 5m --horizon 3
```

---

## 8. أين تجد مخرجات الرسوم البيانية؟

عند اكتمال أي أمر تنبؤ (بدون `--no-charts`)، ستجد ملفات الشارت في مجلد `artifacts/figures/`:
1. **المخطط التفاعلي (Interactive HTML):**  
   `artifacts/figures/<SYMBOL>_<INTERVAL>_forecast_projection_<HORIZON>.html`  
   *(يمكنك فتحه في أي متصفح والتكبير وفحص الشموع والمؤشرات)*
2. **المخطط البياني الثابت (PNG Image):**  
   `artifacts/figures/<SYMBOL>_<INTERVAL>_forecast_projection_<HORIZON>.png`
