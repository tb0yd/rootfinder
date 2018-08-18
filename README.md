# Arabic Rootfinder

A fun little project to play with Jupyter Notebooks, Scikit-learn, and neural nets with Keras.

[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run?template=https://github.com/tb0yd/rootfinder)

#### Goal

To train a neural network to learn Arabic morphology.

#### Includes:

* Scripts for data mining
* Starter data
* 3 iterations of the model: roots.py.ipynb, roots-latin.py.ipynb and roots-with-noroots.py.ipynb.

`roots-with-noroots.py.ipynb` is named weird, but it just means we are more intelligent about tracking words that are "mabniyy", or undeclined.

It's not very accurate (about 50%) so it's pretty addictive to work on. Surely, someone, somewhere, has done this better, but we aren't solving world hunger here, just having some nerdy fun.

Pull requests welcome :)

#### Sample output

The below output is from roots-latin.py.ipynb, which has Latinization in the display stage in Buckwalter mode.

```
Missed.  Input: التراجع (AltrAjE).     Answer: رجع (rjE).    Predicted: ررع (rrE).
Missed.  Input: المركزية (Almrkzyp).   Answer: ركز (rkz).    Predicted: ررك (rrk).
Missed.  Input: القوائم (AlqwA}m).     Answer: قوم (qwm).    Predicted: ققم (qqm).
Missed.  Input: متسلسلة (mtslslp).     Answer: سلسل (slsl).  Predicted: سسلل (ssll).
Correct. Input: جاسوس (jAsws).         Answer: جسس (jss).
Correct. Input: مجدداً (mjddAF).       Answer: جدد (jdd).
Correct. Input: تنْويت (tnowyt).       Answer: نوت (nwt).
Missed.  Input: المحتوى (AlmHtwY).     Answer: حوا (HwA).    Predicted: حوح (HwH).
Correct. Input: رزمة (rzmp).           Answer: رزم (rzm).
Correct. Input: حامل (HAml).           Answer: حمل (Hml).
Correct. Input: يتوسطه (ytwsTh).       Answer: وسط (wsT).
Correct. Input: رفع (rfE).             Answer: رفع (rfE).
Correct. Input: احتياطي (AHtyATy).     Answer: حوط (HwT).
Missed.  Input: الاسم (AlAsm).         Answer: سما (smA).    Predicted: سسم (ssm).
Correct. Input: حرارة (HrArp).         Answer: حرر (Hrr).
Missed.  Input: التزاوج (AltzAwj).     Answer: زوج (zwj).    Predicted: وزز (wzz).
Correct. Input: دُوَلي (duwaly).       Answer: دول (dwl).
Missed.  Input: لمورد (lmwrd).         Answer: ورد (wrd).    Predicted: ممر (mmr).
Correct. Input: إحصاء (<HSA').         Answer: حصا (HSA).
Missed.  Input: كمومي (kmwmy).         Answer:  ().          Predicted: كمم (kmm).
Correct. Input: قائمة (qA}mp).         Answer: قوم (qwm).
Correct. Input: مراسم (mrAsm).         Answer: رسم (rsm).
Correct. Input: هاتف (hAtf).           Answer: هتف (htf).
Correct. Input: مطمور (mTmwr).         Answer: طمر (Tmr).
Correct. Input: رقعة (rqEp).           Answer: رقع (rqE).
Correct. Input: حاسوبي (HAswby).       Answer: حسب (Hsb).
Missed.  Input: للترقية (lltrqyp).     Answer: رقا (rqA).    Predicted: لقق (lqq).
Missed.  Input: التوثيق (Altwvyq).     Answer: وثق (wvq).    Predicted: ووق (wwq).
Correct. Input: رئيسية (r}ysyp).       Answer: رءس (r's).
Missed.  Input: مستطيل (mstTyl).       Answer: طول (Twl).    Predicted: سول (swl).
Missed.  Input: ملف (mlf).             Answer: لفف (lff).    Predicted: ففف (fff).
Missed.  Input: مُقتطف (muqtTf).       Answer: قطف (qTf).    Predicted: ققط (qqT).
Correct. Input: خارطة (xArTp).         Answer: خرط (xrT).
Correct. Input: بالنفاذ (bAlnfA*).     Answer: نفذ (nf*).
Missed.  Input: المعطيات (AlmETyAt).   Answer: عطا (ETA).    Predicted: عطع (ETE).
Missed.  Input: الموعد (AlmwEd).       Answer: وعد (wEd).    Predicted: ومد (wmd).
Correct. Input: توفر (twfr).           Answer: وفر (wfr).
Correct. Input: مربع (mrbE).           Answer: ربع (rbE).
Correct. Input: عمليات (EmlyAt).       Answer: عمل (Eml).
Correct. Input: متصل (mtSl).           Answer: وصل (wSl).
Correct. Input: أعداد (>EdAd).         Answer: عدد (Edd).
Missed.  Input: نصفية (nSfyp).         Answer: نصف (nSf).    Predicted: نفو (nfw).
Missed.  Input: زرً (zrF).             Answer: زرر (zrr).    Predicted: زرا (zrA).
Correct. Input: مشروط (m$rwT).         Answer: شرط ($rT).
Missed.  Input: مكافِئ (mkAfi}).       Answer: كفء (kf').    Predicted: ككف (kkf).
Missed.  Input: يدمج (ydmj).           Answer: دمج (dmj).    Predicted: دمم (dmm).
Missed.  Input: طوب (Twb).             Answer: طوب (Twb).    Predicted: طبب (Tbb).
Correct. Input: مواصلات (mwASlAt).     Answer: وصل (wSl).
Correct. Input: طِبْقُ (Tiboqu).       Answer: طبق (Tbq).
Correct. Input: وراثية (wrAvyp).       Answer: ورث (wrv).
Missed.  Input: ضغط (DgT).             Answer: ضغط (DgT).    Predicted: ضطط (DTT).
Missed.  Input: الإنترنت (Al<ntrnt).   Answer:  ().          Predicted: دون (dwn).
Missed.  Input: الأشغال (Al>$gAl).     Answer: شغل ($gl).    Predicted: ششل ($$l).
Correct. Input: الأخبار (Al>xbAr).     Answer: خبر (xbr).
Correct. Input: تعريفات (tEryfAt).     Answer: عرف (Erf).
Correct. Input: الذاتي (Al*Aty).       Answer: ذوت (*wt).
Missed.  Input: لليمين (llymyn).       Answer: يمن (ymn).    Predicted: لين (lyn).
Correct. Input: إطلاق (<TlAq).         Answer: طلق (Tlq).
Missed.  Input: التفرّع (Altfr~E).     Answer: فرع (frE).    Predicted: قفع (qfE).
Missed.  Input: بُنية (bunyp).         Answer: بنا (bnA).    Predicted: بنن (bnn).
Missed.  Input: مكافئة (mkAf}p).       Answer: كفء (kf').    Predicted: كفف (kff).
Correct. Input: التطبيق (AltTbyq).     Answer: طبق (Tbq).
Correct. Input: توسيع (twsyE).         Answer: وسع (wsE).
Correct. Input: الإرسال (Al<rsAl).     Answer: رسل (rsl).
Missed.  Input: خلفية (xlfyp).         Answer: خلف (xlf).    Predicted: خلو (xlw).
Missed.  Input: الغابة (AlgAbp).       Answer: غيب (gyb).    Predicted: لبب (lbb).
Correct. Input: كسولة (kswlp).         Answer: كسل (ksl).
Correct. Input: عمودي (Emwdy).         Answer: عمد (Emd).
Correct. Input: يضاعف (yDAEf).         Answer: ضعف (DEf).
Correct. Input: خطاط (xTAT).           Answer: خطط (xTT).
Missed.  Input: متناهي (mtnAhy).       Answer: نهو (nhw).    Predicted: ننه (nnh).
Correct. Input: دقيقة (dqyqp).         Answer: دقق (dqq).
Correct. Input: مضافة (mDAfp).         Answer: ضيف (Dyf).
Correct. Input: الخدمة (Alxdmp).       Answer: خدم (xdm).
Missed.  Input: أو (>w).               Answer:  ().          Predicted: ووا (wwA).
Correct. Input: بديل (bdyl).           Answer: بدل (bdl).
Correct. Input: فوريه (fwryh).         Answer: فور (fwr).
Correct. Input: استمارة (AstmArp).     Answer: مور (mwr).
Correct. Input: عيًنة (EyFnp).         Answer: عين (Eyn).
Correct. Input: استخرج (Astxrj).       Answer: خرج (xrj).
Missed.  Input: الفحص (AlfHS).         Answer: فحص (fHS).    Predicted: ففف (fff).
Missed.  Input: الجلسة (Aljlsp).       Answer: جلس (jls).    Predicted: ججل (jjl).
Missed.  Input: باعث (bAEv).           Answer: بعث (bEv).    Predicted: بعع (bEE).
Missed.  Input: الانترنت (AlAntrnt).   Answer:  ().          Predicted: نون (nwn).
Missed.  Input: تناوبي (tnAwby).       Answer: نوب (nwb).    Predicted: ننب (nnb).
Correct. Input: تكرار (tkrAr).         Answer: كرر (krr).
Correct. Input: طول (Twl).             Answer: طول (Twl).
Missed.  Input: اكتشاف (Akt$Af).       Answer: كشف (k$f).    Predicted: ككف (kkf).
Missed.  Input: مع (mE).               Answer:  ().          Predicted: ععع (EEE).
Correct. Input: برمجية (brmjyp).       Answer: برمج (brmj).
Correct. Input: عُشري (Eu$ry).         Answer: عشر (E$r).
Missed.  Input: برتقالي (brtqAly).     Answer:  ().          Predicted: بقل (bql).
Correct. Input: زيارة (zyArp).         Answer: زور (zwr).
Correct. Input: تأشير (t>$yr).         Answer: اشر (A$r).
Missed.  Input: معالجة (mEAljp).       Answer: علج (Elj).    Predicted: علل (Ell).
Correct. Input: سرقة (srqp).           Answer: سرق (srq).
Missed.  Input: منتج (mntj).           Answer: نتج (ntj).    Predicted: نجج (njj).
Missed.  Input: ترقية (trqyp).         Answer: رقا (rqA).    Predicted: رقق (rqq).
Missed.  Input: البديهية (Albdyhyp).   Answer: بده (bdh).    Predicted: بدد (bdd).
Missed.  Input: المراجع (AlmrAjE).     Answer: رجع (rjE).    Predicted: ررع (rrE).
Missed.  Input: إيقاف (<yqAf).         Answer: وقف (wqf).    Predicted: ويف (wyf).
Correct. Input: لافتة (lAftp).         Answer: لفت (lft).
Correct. Input: حلقة (Hlqp).           Answer: حلق (Hlq).
Correct. Input: مراجعة (mrAjEp).       Answer: رجع (rjE).
Correct. Input: عكسية (Eksyp).         Answer: عكس (Eks).
Correct. Input: مخطط (mxTT).           Answer: خطط (xTT).
Correct. Input: أقراص (>qrAS).         Answer: قرص (qrS).
Correct. Input: مشهد (m$hd).           Answer: شهد ($hd).
Missed.  Input: العطب (AlETb).         Answer: عطب (ETb).    Predicted: عطع (ETE).
Correct. Input: خفيفة (xfyfp).         Answer: خفف (xff).
Missed.  Input: مبني (mbny).           Answer: بنا (bnA).    Predicted: بنن (bnn).
Missed.  Input: حركي (Hrky).           Answer: حرك (Hrk).    Predicted: حكك (Hkk).
Correct. Input: مكيف (mkyf).           Answer: كيف (kyf).
Missed.  Input: انصهار (AnShAr).       Answer: صهر (Shr).    Predicted: نصر (nSr).
Correct. Input: تهليل (thlyl).         Answer: هلل (hll).
Missed.  Input: إيداع (<ydAE).         Answer: ودع (wdE).    Predicted: ويع (wyE).
Correct. Input: الألفية (Al>lfyp).     Answer: الف (Alf).
Missed.  Input: أولا (>wlA).           Answer: اول (Awl).    Predicted: الل (All).
Correct. Input: فرع (frE).             Answer: فرع (frE).
Correct. Input: الذاكرة (Al*Akrp).     Answer: ذكر (*kr).
Correct. Input: رأسي (r>sy).           Answer: رءس (r's).
Missed.  Input: مواءمة (mwA'mp).       Answer: وءم (w'm).    Predicted: ووم (wwm).
Missed.  Input: الاموال (AlAmwAl).     Answer: مول (mwl).    Predicted: ومل (wml).
Correct. Input: جانب (jAnb).           Answer: جنب (jnb).
Correct. Input: احتقان (AHtqAn).       Answer: حقن (Hqn).
Missed.  Input: حقل (Hql).             Answer: حقل (Hql).    Predicted: حلل (Hll).
Missed.  Input: قتلة (qtlp).           Answer: قتل (qtl).    Predicted: قلل (qll).
Correct. Input: جدار (jdAr).           Answer: جدر (jdr).
Correct. Input: نافذة (nAf*p).         Answer: نفذ (nf*).
Missed.  Input: البوصة (AlbwSp).       Answer:  ().          Predicted: وصب (wSb).
Correct. Input: الصورة (AlSwrp).       Answer: صور (Swr).
Missed.  Input: هدف (hdf).             Answer: هدف (hdf).    Predicted: هفف (hff).
Correct. Input: مركَّب (mrka~b).       Answer: ركب (rkb).
Correct. Input: دوري (dwry).           Answer: دور (dwr).
Missed.  Input: السّطور (Als~Twr).     Answer: سطر (sTr).    Predicted: سسط (ssT).
Missed.  Input: غلط (glT).             Answer: غلط (glT).    Predicted: غلو (glw).
Score: 55.9%
```

#### How to use

Just install Jupyter Notebook and run `jupyter notebook` in this folder, and select one of the `ipynb` files.

Or click this button to open JupyterLab workspace on FloydHub:
[![Run on FloydHub](https://static.floydhub.com/button/button-small.svg)](https://floydhub.com/run?template=https://github.com/tb0yd/rootfinder)
