# Arabic Rootfinder

A fun little project to play with Jupyter Notebooks, Scikit-learn, and neural nets with Keras.

#### Goal

To train a neural network to learn Arabic morphology.

#### Includes:

* Scripts for data mining
* Starter data
* 2 iterations of the model: roots.ipynb and roots-with-noroots.ipynb.

`roots-with-noroots.ipynb` is named weird, but it just means we are more intelligent about tracking words that are "mabniyy", or undeclined.

It's not very accurate (about 50%) so it's pretty addictive to work on. Surely, someone, somewhere, has done this better, but we aren't solving world hunger here, just having some nerdy fun.

Pull requests welcome :)

#### Sample output

The output is formatted as "accuracy: [correctAnswer, input]" with output on the end if incorrect.

```
Missed:  ['مادي', 'مدد'] Predicted: مدو
Missed:  ['برتقالي', ''] Predicted: رقل
Missed:  ['تسلسلي', 'سلسل'] Predicted: سلل
Correct: ['وسيط', 'وسط']
Correct: ['مرشح', 'رشح']
Correct: ['تفعيل', 'فعل']
Missed:  ['اختيار', 'خير'] Predicted: خور
Correct: ['جمع', 'جمع']
Missed:  ['الأداء', 'ادي'] Predicted: ادو
Missed:  ['لليسار', 'يسر'] Predicted: لير
Missed:  ['نجدة', 'نجد'] Predicted: ندو
Correct: ['مخلوع', 'خلع']
Missed:  ['الترتيب', 'رتب'] Predicted: ربب
Missed:  ['للزلق', 'زلق'] Predicted: للق
Correct: ['سببي', 'سبب']
Missed:  ['بالحاسوب', 'حسب'] Predicted: حوح
Correct: ['طور', 'طور']
Correct: ['جذري', 'جذر']
Correct: ['توازن', 'وزن']
Correct: ['ثابتة', 'ثبت']
Correct: ['خارجية', 'خرج']
Correct: ['نُسْخَةٌ', 'نسخ']
Missed:  ['نكهة', 'نكه'] Predicted: نهر
Correct: ['تصليح', 'صلح']
Missed:  ['المشترك', 'شرك'] Predicted: ششر
Correct: ['قاسم', 'قسم']
Missed:  ['عزل', 'عزل'] Predicted: علل
Missed:  ['مُدخل', 'دخل'] Predicted: ددل
Missed:  ['الإنجاز', 'نجز'] Predicted: نجج
Missed:  ['متقطع', 'قطع'] Predicted: ققع
Correct: ['يكرر', 'كرر']
Missed:  ['متوفر', 'وفر'] Predicted: وفف
Correct: ['شعاعية', 'شعع']
Missed:  ['هوجاء', 'هوج'] Predicted: وجا
Missed:  ['النمط', 'نمط'] Predicted: ننط
Missed:  ['مُسَمىّ', 'سما'] Predicted: سسم
Correct: ['عكس', 'عكس']
Correct: ['دافع', 'دفع']
Missed:  ['ضغط', 'ضغط'] Predicted: ضطط
Correct: ['مغلف', 'غلف']
Missed:  ['مَخبأ', 'خبء'] Predicted: خخب
Correct: ['ضبابي', 'ضبب']
Correct: ['محوري', 'حور']
Missed:  ['التصليح', 'صلح'] Predicted: صصل
Correct: ['مأزق', 'ازق']
Correct: ['حقيقياً', 'حقق']
Correct: ['طريق', 'طرق']
Missed:  ['نواة', 'نوا'] Predicted: نوو
Missed:  ['شهر', 'شهر'] Predicted: شرر
Missed:  ['رسائل', 'رسل'] Predicted: رءل
Correct: ['حروف', 'حرف']
Correct: ['مكيف', 'كيف']
Missed:  ['مَضْروب', 'ضرب'] Predicted: ضضر
Missed:  ['البوصة', ''] Predicted: وبب
Correct: ['النشاط', 'نشط']
Missed:  ['ملايين', ''] Predicted: مين
Missed:  ['إِدَانَة', 'دين'] Predicted: ودن
Missed:  ['البرمجيات', 'برمج'] Predicted: ربم
Missed:  ['تلفزيونيّة', ''] Predicted: وفا
Missed:  ['مفاتيح', 'فتح'] Predicted: ففح
Correct: ['إجراءات', 'جرا']
Correct: ['صدّ', 'صدد']
Missed:  ['باليد', 'يدد'] Predicted: بدد
Missed:  ['أزمة', 'ازم'] Predicted: ززم
Correct: ['عشوائية', 'عشو']
Missed:  ['مستمرة', 'مرر'] Predicted: مرم
Correct: ['الخطأ', 'خطا']
Correct: ['مُعامل', 'عمل']
Correct: ['القرارات', 'قرر']
Correct: ['مغلق', 'غلق']
Correct: ['إشباع', 'شبع']
Correct: ['بطيء', 'بطء']
Correct: ['انتشار', 'نشر']
Correct: ['يعمل', 'عمل']
Correct: ['مضافة', 'ضيف']
Correct: ['نيابة', 'نوب']
Correct: ['دَخْل', 'دخل']
Missed:  ['الأدنى', 'دنو'] Predicted: ادو
Missed:  ['تعويذة', 'عوذ'] Predicted: عوا
Missed:  ['سجل', 'سجل'] Predicted: سلل
Correct: ['عالية', 'علو']
Correct: ['مكتبة', 'كتب']
Correct: ['إدراج', 'درج']
Correct: ['تقرير', 'قرر']
Missed:  ['ذكي', 'ذكا'] Predicted: ككر
Correct: ['تقريباً', 'قرب']
Correct: ['التوصيلات', 'وصل']
Correct: ['طرفيّة', 'طرف']
Missed:  ['تَنويت', 'نوت'] Predicted: نوا
Missed:  ['عقد', 'عقد'] Predicted: عدد
Correct: ['انضمامي', 'ضمم']
Correct: ['مثالي', 'مثل']
Missed:  ['مسيًر', 'سير'] Predicted: سسر
Correct: ['قانونية', 'قنن']
Correct: ['ذاتي', 'ذوت']
Correct: ['الشكل', 'شكل']
Correct: ['مشترك', 'شرك']
Correct: ['مقهى', 'قهو']
Missed:  ['تطويع', 'طوع'] Predicted: ططع
Missed:  ['داليّة', 'دلو'] Predicted: دلل
Missed:  ['أو', ''] Predicted: اوو
Correct: ['نداء', 'ندو']
Missed:  ['ديناميكياً', ''] Predicted: دون
Missed:  ['بصر', 'بصر'] Predicted: صرر
Correct: ['مجلد', 'جلد']
Correct: ['إيقاف', 'وقف']
Missed:  ['الاستفهام', 'فهم'] Predicted: سلف
Correct: ['قواعد', 'قعد']
Correct: ['إقلاع', 'قلع']
Correct: ['مصفوفة', 'صفف']
Missed:  ['استطلاع', 'طلع'] Predicted: سطع
Correct: ['نقطتان', 'نقط']
Correct: ['استعداد', 'عدد']
Missed:  ['ظرف', 'ظرف'] Predicted: ررف
Missed:  ['مُجمَل', 'جمل'] Predicted: ممل
Correct: ['جلب', 'جلب']
Correct: ['متصلة', 'وصل']
Missed:  ['الإطار', 'اطر'] Predicted: ططر
Correct: ['تزايد', 'زيد']
Missed:  ['أعظم', 'عظم'] Predicted: عمم
Correct: ['متجانس', 'جنس']
Correct: ['معدّل', 'عدل']
Correct: ['تقصّص', 'قصص']
Missed:  ['دي', ''] Predicted: دور
Correct: ['صور', 'صور']
Correct: ['خفيفة', 'خفف']
Missed:  ['حَدسي', 'حدس'] Predicted: حدد
Correct: ['شريط', 'شرط']
Correct: ['المستوى', 'سوا']
Correct: ['واسعة', 'وسع']
Missed:  ['المتعددة', 'عدد'] Predicted: ودد
Correct: ['أعيان', 'عين']
Correct: ['عامة', 'عمم']
Correct: ['منفعة', 'نفع']
Missed:  ['حجاب', 'حجب'] Predicted: حبب
Missed:  ['التحديث', 'حدث'] Predicted: حدد
Score: 56.6%
```

#### How to use

Just install Jupyter Notebook and run `jupyter notebook` in this folder, and select one of the `ipynb` files.
