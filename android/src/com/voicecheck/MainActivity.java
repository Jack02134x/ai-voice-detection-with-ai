package com.voicecheck;

import android.Manifest;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Switch;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** One screen: check a shared / chosen recording, or listen live. Gauge, waveform, P(AI) timeline. */
public class MainActivity extends Activity {
    static final int BG = 0xFF070A12, CARD = 0xFF111829, LINE = 0xFF1F2A40, TEXT = 0xFFE8EDF7,
            MUTED = 0xFF8B97B0, HUMAN = 0xFF2DD4BF, AI = 0xFFFB4F6F, TILE = 0xFF0B1120;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private Analyzer analyzer;
    private double threshold = 0.45;
    private GaugeView gauge;
    private WaveformView wave;
    private TimelineView timeline;
    private TextView verdict, detail, barHv, barAv, status, foot, thrVal;
    private View barH, barA;
    private View waveCard;
    private ProgressBar level;
    private Switch recSwitch;
    private TextView recLabel;
    private SeekBar thrSeek;
    private Button checkBtn, liveBtn;
    private final TextView[] tileLabel = new TextView[4], tileValue = new TextView[4];
    private float dp;

    // current displayed verdict (so the threshold slider can re-evaluate it)
    private double curP = Double.NaN;
    private boolean curNoSpeech = false;

    // live state (the LiveService owns the mic; the activity just renders)
    private boolean liveUi = false;
    private Runnable frameTick;

    // file state
    private float[] filePerWindow = new float[0];

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        dp = getResources().getDisplayMetrics().density;
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);
        setContentView(build());
        new Thread(this::loadModel).start();
        rescueOrphanRecordings();
    }

    private void loadModel() {
        try {
            analyzer = new Analyzer(this);
            threshold = analyzer.threshold;
            ui.post(() -> {
                thrSeek.setProgress((int) Math.round(threshold * 100));
                gauge.set(Double.NaN, threshold);
                timeline.setThreshold(threshold);
                verdict.setText("READY");
                foot.setText("Model: " + analyzer.source + " · runs on this phone");
                handleIntent(getIntent());
            });
        } catch (Exception e) {
            ui.post(() -> status.setText("Could not load model: " + e.getMessage()));
        }
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (analyzer != null) handleIntent(intent);
    }

    private void handleIntent(Intent intent) {
        if (intent == null) return;
        Uri uri = null;
        if (Intent.ACTION_SEND.equals(intent.getAction())) uri = intent.getParcelableExtra(Intent.EXTRA_STREAM);
        else if (Intent.ACTION_VIEW.equals(intent.getAction())) uri = intent.getData();
        if (uri != null) checkFile(uri);
    }

    // ---------------- layout ----------------

    private View build() {
        ScrollView scroll = new ScrollView(this);
        scroll.setBackgroundColor(BG);
        scroll.setFillViewport(true);
        LinearLayout root = col();
        root.setPadding(px(18), px(24), px(18), px(24));
        scroll.addView(root);

        root.addView(text("Voice Check", 26, TEXT, true));
        TextView sub = text("Is this voice real or AI-generated?", 14, MUTED, false);
        sub.setPadding(0, px(2), 0, px(14));
        root.addView(sub);

        LinearLayout card = card();
        card.setGravity(Gravity.CENTER_HORIZONTAL);
        gauge = new GaugeView(this);
        card.addView(gauge, new LinearLayout.LayoutParams(px(230), px(230)));
        verdict = text("LOADING…", 29, MUTED, true);
        verdict.setGravity(Gravity.CENTER);
        card.addView(verdict);
        detail = text("", 13, MUTED, false);
        detail.setGravity(Gravity.CENTER);
        detail.setPadding(0, px(4), 0, px(12));
        card.addView(detail);
        card.addView(bar("HUMAN", HUMAN, true));
        card.addView(bar("AI", AI, false));
        root.addView(card);

        LinearLayout btns = new LinearLayout(this);
        btns.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams bm = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, px(54));
        bm.setMargins(0, px(12), 0, px(4));
        checkBtn = button("Check a recording", HUMAN, 0xFF061016);
        checkBtn.setOnClickListener(v -> pickFile());
        liveBtn = button("Listen live", 0xFF1A2338, TEXT);
        liveBtn.setOnClickListener(v -> toggleLive());
        LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        half.setMargins(0, 0, px(5), 0);
        btns.addView(checkBtn, half);
        LinearLayout.LayoutParams half2 = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1);
        half2.setMargins(px(5), 0, 0, 0);
        btns.addView(liveBtn, half2);
        root.addView(btns, bm);

        level = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        level.setMax(100);
        level.setProgressTintList(android.content.res.ColorStateList.valueOf(HUMAN));
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, px(7));
        lp.setMargins(0, px(8), 0, 0);
        root.addView(level, lp);

        // save live recordings toggle + REC indicator
        LinearLayout recRow = new LinearLayout(this);
        recRow.setOrientation(LinearLayout.HORIZONTAL);
        recRow.setGravity(Gravity.CENTER_VERTICAL);
        recRow.setPadding(0, px(10), 0, 0);
        recSwitch = new Switch(this);
        recSwitch.setText("Save live recordings");
        recSwitch.setTextColor(TEXT);
        recSwitch.setTextSize(14);
        recSwitch.setChecked(Recordings.enabled(this));
        recSwitch.setThumbTintList(android.content.res.ColorStateList.valueOf(AI));
        recSwitch.setOnCheckedChangeListener((b, on) -> setRecording(on));
        recRow.addView(recSwitch, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        recLabel = text("", 13, AI, true);
        recRow.addView(recLabel);
        root.addView(recRow);
        TextView recHint = text("When you stop, you'll be asked if the voice was human or AI; the WAV goes to "
                + Recordings.visibleFolder() + "/human, /ai, or the folder itself if unsure.", 11, MUTED, false);
        recHint.setPadding(0, px(2), 0, 0);
        root.addView(recHint);

        // waveform (live input)
        waveCard = card();
        ((LinearLayout) waveCard).addView(header("LIVE INPUT"));
        wave = new WaveformView(this);
        ((LinearLayout) waveCard).addView(wave, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, px(90)));
        waveCard.setVisibility(View.GONE);
        LinearLayout.LayoutParams wm = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        wm.setMargins(0, px(12), 0, 0);
        root.addView(waveCard, wm);

        // timeline + threshold
        LinearLayout tl = card();
        LinearLayout.LayoutParams tm = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        tm.setMargins(0, px(12), 0, 0);
        tl.addView(header("P(AI) OVER TIME"));
        timeline = new TimelineView(this);
        tl.addView(timeline, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, px(150)));
        LinearLayout thrRow = new LinearLayout(this);
        thrRow.setOrientation(LinearLayout.HORIZONTAL);
        thrRow.setGravity(Gravity.CENTER_VERTICAL);
        thrRow.setPadding(0, px(10), 0, 0);
        thrRow.addView(text("Threshold", 13, MUTED, false));
        thrSeek = new SeekBar(this);
        thrSeek.setMax(95);
        thrSeek.setMin(5);
        thrSeek.setProgress(45);
        thrSeek.getProgressDrawable().setTint(0xFFF5B94A);
        thrSeek.getThumb().setTint(0xFFF5B94A);
        thrSeek.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar s, int v, boolean fromUser) {
                threshold = v / 100.0;
                thrVal.setText(String.format(Locale.US, "%.2f", threshold));
                reapplyThreshold();
            }
            public void onStartTrackingTouch(SeekBar s) {}
            public void onStopTrackingTouch(SeekBar s) {}
        });
        LinearLayout.LayoutParams sl = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
        sl.setMargins(px(8), 0, px(8), 0);
        thrRow.addView(thrSeek, sl);
        thrVal = text("0.45", 13, TEXT, false);
        thrVal.setTypeface(Typeface.MONOSPACE);
        thrRow.addView(thrVal);
        tl.addView(thrRow);
        root.addView(tl, tm);

        // stat tiles (2x2)
        LinearLayout statsTop = statRow(0, 1), statsBot = statRow(2, 3);
        LinearLayout.LayoutParams sm = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        sm.setMargins(0, px(12), 0, 0);
        root.addView(statsTop, sm);
        LinearLayout.LayoutParams sm2 = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        sm2.setMargins(0, px(8), 0, 0);
        root.addView(statsBot, sm2);
        setFileTiles();

        status = text("Share a voice note to Voice Check, or tap a button above.", 13, MUTED, false);
        status.setPadding(0, px(12), 0, 0);
        root.addView(status);
        TextView about = text("Share any voice note or recording to this app, or pick one. \"Listen live\" uses the "
                + "phone mic. Silence and non-speech are detected (Silero) and shown as NO SPEECH. Everything runs "
                + "on the phone. On phone-call / compressed audio the result is a rough guess, not proof.", 12, MUTED, false);
        about.setPadding(px(2), px(14), px(2), px(8));
        root.addView(about);
        foot = text("", 11, 0xFF56627C, false);
        root.addView(foot);
        return scroll;
    }

    private LinearLayout statRow(int a, int b) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.addView(tile(a), tileLp(true));
        row.addView(tile(b), tileLp(false));
        return row;
    }

    private LinearLayout.LayoutParams tileLp(boolean left) {
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1);
        p.setMargins(left ? 0 : px(4), 0, left ? px(4) : 0, 0);
        return p;
    }

    private LinearLayout tile(int i) {
        LinearLayout t = col();
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(TILE);
        bg.setCornerRadius(px(12));
        bg.setStroke(px(1), LINE);
        t.setBackground(bg);
        t.setPadding(px(12), px(10), px(12), px(10));
        tileLabel[i] = text("", 11, MUTED, false);
        tileLabel[i].setLetterSpacing(0.08f);
        tileValue[i] = text("—", 19, TEXT, false);
        tileValue[i].setTypeface(Typeface.MONOSPACE);
        tileValue[i].setPadding(0, px(3), 0, 0);
        t.addView(tileLabel[i]);
        t.addView(tileValue[i]);
        return t;
    }

    private void setFileTiles() {
        tileLabel[0].setText("DURATION");
        tileLabel[1].setText("SPEECH WINDOWS");
        tileLabel[2].setText("AI WINDOWS");
        tileLabel[3].setText("ANALYSIS TIME");
    }

    private void setLiveTiles() {
        tileLabel[0].setText("SESSION P(AI)");
        tileLabel[1].setText("SPEECH WINDOWS");
        tileLabel[2].setText("AI WINDOWS");
        tileLabel[3].setText("LISTENING");
    }

    private View bar(String name, int color, boolean human) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, px(3), 0, px(3));
        TextView n = text(name, 12, color, true);
        n.setWidth(px(58));
        row.addView(n);
        LinearLayout tr = new LinearLayout(this);
        GradientDrawable trbg = new GradientDrawable();
        trbg.setColor(0xFF1A2338);
        trbg.setCornerRadius(px(5));
        tr.setBackground(trbg);
        View fill = new View(this);
        GradientDrawable fbg = new GradientDrawable();
        fbg.setColor(color);
        fbg.setCornerRadius(px(5));
        fill.setBackground(fbg);
        tr.addView(fill, new LinearLayout.LayoutParams(0, px(9)));
        LinearLayout.LayoutParams trp = new LinearLayout.LayoutParams(0, px(9), 1);
        trp.setMargins(px(6), 0, px(8), 0);
        row.addView(tr, trp);
        TextView val = text("—", 12, TEXT, false);
        val.setWidth(px(52));
        val.setGravity(Gravity.END);
        row.addView(val);
        if (human) { barH = fill; barHv = val; } else { barA = fill; barAv = val; }
        return row;
    }

    private LinearLayout col() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    private LinearLayout card() {
        LinearLayout c = col();
        GradientDrawable bg = new GradientDrawable();
        bg.setColor(CARD);
        bg.setCornerRadius(px(18));
        bg.setStroke(px(1), LINE);
        c.setBackground(bg);
        c.setPadding(px(16), px(14), px(16), px(14));
        return c;
    }

    private TextView header(String s) {
        TextView t = text(s, 11, MUTED, true);
        t.setLetterSpacing(0.1f);
        t.setPadding(0, 0, 0, px(8));
        return t;
    }

    private TextView text(String s, int sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(sp);
        t.setTextColor(color);
        if (bold) t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    private Button button(String s, int bg, int fg) {
        Button b = new Button(this);
        b.setText(s);
        b.setAllCaps(false);
        b.setTextSize(15);
        b.setTypeface(Typeface.DEFAULT_BOLD);
        b.setTextColor(fg);
        GradientDrawable d = new GradientDrawable();
        d.setColor(bg);
        d.setCornerRadius(px(14));
        b.setBackground(d);
        b.setStateListAnimator(null);
        b.setPadding(0, 0, 0, 0);
        return b;
    }

    private int px(float v) {
        return Math.round(v * dp);
    }

    // ---------------- verdict / threshold ----------------

    private void showVerdict(double p, boolean noSpeech) {
        curP = p;
        curNoSpeech = noSpeech;
        gauge.set(noSpeech ? Double.NaN : p, threshold);
        if (noSpeech || Double.isNaN(p)) {
            barH.getLayoutParams().width = 0;
            barA.getLayoutParams().width = 0;
            barH.requestLayout();
            barA.requestLayout();
            barHv.setText("—");
            barAv.setText("—");
            return;
        }
        boolean ai = p >= threshold;
        verdict.setText(ai ? "AI-GENERATED" : "HUMAN");
        verdict.setTextColor(ai ? AI : HUMAN);
        final double pp = p;
        ((View) barH.getParent()).post(() -> {
            int w = ((View) barH.getParent()).getWidth();
            barH.getLayoutParams().width = (int) (w * (1 - pp));
            barA.getLayoutParams().width = (int) (w * pp);
            barH.requestLayout();
            barA.requestLayout();
        });
        barHv.setText(String.format(Locale.US, "%.1f%%", (1 - p) * 100));
        barAv.setText(String.format(Locale.US, "%.1f%%", p * 100));
    }

    /** Re-evaluate the current verdict, timeline colouring and AI-window count at the new threshold. */
    private void reapplyThreshold() {
        timeline.setThreshold(threshold);
        if (!curNoSpeech && !Double.isNaN(curP)) showVerdict(curP, false);
        else gauge.set(Double.NaN, threshold);
        if (liveUi && LiveService.instance != null) {
            tileValue[2].setText(aiWindowPct(LiveService.instance.session));
        } else if (filePerWindow.length > 0) {
            List<Float> l = new ArrayList<>();
            for (float f : filePerWindow) l.add(f);
            tileValue[2].setText(aiWindowPct(l));
        }
    }

    private String aiWindowPct(List<Float> scores) {
        if (scores.isEmpty()) return "—";
        int ai = 0;
        for (float s : scores) if (s >= threshold) ai++;
        return Math.round(ai * 100f / scores.size()) + "%";
    }

    // ---------------- file check ----------------

    private void pickFile() {
        Intent i = new Intent(Intent.ACTION_GET_CONTENT);
        i.setType("*/*");
        i.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"audio/*", "video/*"});
        i.addCategory(Intent.CATEGORY_OPENABLE);
        try {
            startActivityForResult(Intent.createChooser(i, "Pick a recording"), 1);
        } catch (Exception e) {
            toast("No file picker available");
        }
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == 1 && res == RESULT_OK && data != null && data.getData() != null) checkFile(data.getData());
    }

    private void checkFile(Uri uri) {
        if (analyzer == null) {
            toast("Model still loading…");
            return;
        }
        if (liveUi) stopLive();
        setFileTiles();
        waveCard.setVisibility(View.GONE);
        timeline.setSpan(0);
        timeline.clear();
        showVerdict(Double.NaN, false);
        verdict.setText("ANALYSING…");
        verdict.setTextColor(MUTED);
        detail.setText("");
        status.setText("Decoding and scoring…");
        new Thread(() -> {
            try {
                float[] wav = Decode.file(this, uri);
                long t = SystemClock.elapsedRealtime();
                Analyzer.Result r = analyzer.analyze(wav);
                int ms = (int) (SystemClock.elapsedRealtime() - t);
                float dur = wav.length / (float) Preprocess.SR;
                ui.post(() -> showFileResult(r, r.starts, dur, ms));
            } catch (Exception e) {
                ui.post(() -> {
                    verdict.setText("ERROR");
                    verdict.setTextColor(AI);
                    status.setText(String.valueOf(e.getMessage()));
                });
            }
        }).start();
    }

    private void showFileResult(Analyzer.Result r, double[] starts, float dur, int ms) {
        tileValue[0].setText(fmtTime(dur));
        tileValue[3].setText(ms + " ms");
        if (r.noSpeech) {
            showVerdict(Double.NaN, true);
            verdict.setText("NO SPEECH");
            verdict.setTextColor(MUTED);
            detail.setText("No speech found");
            tileValue[1].setText("0");
            tileValue[2].setText("—");
            filePerWindow = new float[0];
            timeline.clear();
        } else {
            filePerWindow = r.perWindow;
            showVerdict(r.pAi, false);
            detail.setText(String.format(Locale.US, "%.1f s · %d speech window%s", dur, r.windows, r.windows == 1 ? "" : "s"));
            tileValue[1].setText(String.valueOf(r.windows));
            List<Float> l = new ArrayList<>();
            for (float f : r.perWindow) l.add(f);
            tileValue[2].setText(aiWindowPct(l));
            List<double[]> pts = new ArrayList<>();
            for (int i = 0; i < r.perWindow.length; i++) pts.add(new double[]{starts.length > i ? starts[i] : i, r.perWindow[i]});
            timeline.set(pts, threshold);
        }
        status.setText("Done.");
    }

    // ---------------- live mic (runs in LiveService so it survives minimising) ----------------

    private void toggleLive() {
        if (liveUi) {
            stopLive();
            return;
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 2);
            return;
        }
        if (!android.provider.Settings.canDrawOverlays(this)) {    // for the floating pill when minimised
            toast("Allow \"Display over other apps\" so the % can float when minimised");
            try {
                startActivity(new Intent(android.provider.Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName())));
            } catch (Exception ignored) {
            }
            // still start listening; the pill just won't show until the permission is granted
        }
        startLive();
    }

    private void startLive() {
        LiveService.start(this);
        enterLiveUi();
    }

    private void enterLiveUi() {
        liveUi = true;
        setLiveTiles();
        timeline.setSpan(30);
        timeline.clear();
        waveCard.setVisibility(View.VISIBLE);
        liveBtn.setText("Stop");
        liveBtn.setTextColor(Color.WHITE);
        ((GradientDrawable) liveBtn.getBackground()).setColor(AI);
        verdict.setText("LISTENING…");
        verdict.setTextColor(MUTED);
        detail.setText("");
        showVerdict(Double.NaN, true);
        status.setText("Listening to the mic…");
        LiveService.listener = this::renderLive;
        if (LiveService.instance != null) {
            LiveService.instance.setAppVisible(true);
            renderLive();
        }
        frameTick = new Runnable() {
            public void run() {
                if (!liveUi) return;
                LiveService s = LiveService.instance;
                if (s != null && s.mic() != null) {
                    wave.setData(s.mic().recent(Preprocess.SR * 2), true);
                    level.setProgress((int) Math.max(0, Math.min(100, (s.mic().levelDb + 60) / 60 * 100)));
                    tileValue[3].setText(fmtTime((SystemClock.elapsedRealtime() - s.startTime) / 1000f));
                    recLabel.setText(s.recording
                            ? "● REC " + fmtTime((SystemClock.elapsedRealtime() - s.recordingSince) / 1000f) : "");
                }
                ui.postDelayed(this, 80);
            }
        };
        ui.post(frameTick);
    }

    /** Render the service's current state (called on each update, and on resume). */
    private void renderLive() {
        LiveService s = LiveService.instance;
        if (s == null || !liveUi) return;
        List<double[]> pts;
        synchronized (s.points) {
            pts = new ArrayList<>(s.points);
        }
        timeline.set(pts, threshold);
        if (s.noSpeech || Double.isNaN(s.smooth)) {
            verdict.setText("LISTENING… (no speech)");
            verdict.setTextColor(MUTED);
            showVerdict(Double.NaN, true);
        } else {
            showVerdict(s.smooth, false);
            detail.setText(String.format(Locale.US, "Live · %d ms/check", s.lastMs));
        }
        tileValue[0].setText(s.session.isEmpty() ? "—" : String.format(Locale.US, "%.2f", mean(s.session)));
        tileValue[1].setText(String.valueOf(s.session.size()));
        tileValue[2].setText(aiWindowPct(s.session));
    }

    private static double mean(List<Float> l) {
        double s = 0;
        for (float f : l) s += f;
        return s / l.size();
    }

    private String fmtTime(float sec) {
        int s = Math.max(0, Math.round(sec));
        return s / 60 + ":" + String.format(Locale.US, "%02d", s % 60);
    }

    private void stopLive() {
        liveUi = false;
        LiveService.listener = null;
        List<Float> sess = LiveService.instance != null ? LiveService.instance.session : new ArrayList<>();
        LiveService.stop(this);
        liveBtn.setText("Listen live");
        liveBtn.setTextColor(TEXT);
        ((GradientDrawable) liveBtn.getBackground()).setColor(0xFF1A2338);
        level.setProgress(0);
        wave.setData(new float[0], false);
        waveCard.setVisibility(View.GONE);
        status.setText(sess.isEmpty() ? "Stopped." : String.format(Locale.US, "Stopped. Session P(AI) %.2f over %d windows.", mean(sess), sess.size()));
        recLabel.setText("");
        ui.postDelayed(() -> askLabel(0), 600);
    }

    /** The "Save live recordings" switch: remembered, and applied right away if listening. */
    private void setRecording(boolean on) {
        if (on && android.os.Build.VERSION.SDK_INT < 29
                && checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 3);
        }
        Recordings.setEnabled(this, on);
        LiveService s = LiveService.instance;
        if (s != null) {
            if (on) s.startRecording();
            else {
                s.stopRecording(true);
                recLabel.setText("");
                ui.postDelayed(() -> askLabel(0), 600);
            }
        } else {
            toast(on ? "Live sessions will be saved to " + Recordings.visibleFolder() : "Live sessions won't be saved");
        }
    }

    /** After a recorded session: ask whether the voice was human or AI, then file it. */
    private void askLabel(int attempt) {
        LiveService.Pending pr = LiveService.pending;
        if (pr == null) {
            if (attempt < 8) ui.postDelayed(() -> askLabel(attempt + 1), 300);   // service still finishing
            else if (!LiveService.lastSaved.isEmpty()) toast(LiveService.lastSaved);
            return;
        }
        LiveService.pending = null;
        long secs = pr.pcm.length() / (2L * Preprocess.SR);
        String guess = Double.isNaN(pr.meanPAi) ? "the detector heard no speech"
                : String.format(Locale.US, "the detector said %d%% AI", Math.round(pr.meanPAi * 100));
        String[] choices = {"Human", "AI", "Not sure (save unlabelled)", "Discard"};
        new AlertDialog.Builder(this, android.R.style.Theme_Material_Dialog_Alert)
                .setTitle(String.format(Locale.US, "Label this recording (%d s, %s)", secs, guess))
                .setItems(choices, (d, which) -> {
                    if (which == 3) {
                        pr.pcm.delete();
                        toast("Recording discarded");
                    } else {
                        saveLabelled(pr, which == 0 ? "human" : which == 1 ? "ai" : null);
                    }
                })
                .setOnCancelListener(d -> saveLabelled(pr, null))   // back / tap outside: keep it, unlabelled
                .show();
    }

    private void saveLabelled(LiveService.Pending pr, String label) {
        new Thread(() -> {
            String msg;
            try {
                String path = Recordings.save(this, pr.pcm, pr.meanPAi, label);
                msg = path == null ? "Recording under 1 s, not saved" : "Saved to " + path;
            } catch (Exception e) {
                msg = "Saving failed: " + e.getMessage();
            }
            final String m = msg;
            ui.post(() -> {
                status.setText(m);
                toast(m);
            });
        }).start();
    }

    /** Recordings left in the cache by a killed app (never labelled): keep them, unlabelled. */
    private void rescueOrphanRecordings() {
        if (LiveService.instance != null || LiveService.pending != null) return;   // one is in use
        java.io.File[] left = getCacheDir().listFiles((d, n) -> n.startsWith("live_") && n.endsWith(".pcm"));
        if (left == null || left.length == 0) return;
        new Thread(() -> {
            int saved = 0;
            for (java.io.File f : left) {
                try {
                    if (Recordings.save(this, f, Double.NaN, null) != null) saved++;
                } catch (Exception ignored) {
                }
            }
            final int n = saved;
            if (n > 0) ui.post(() -> toast(n + " unfinished recording(s) saved to " + Recordings.visibleFolder()));
        }).start();
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] p, int[] r) {
        if (code == 2 && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) toggleLive();
        else toast("Microphone permission is needed for live mode");
    }

    @Override
    protected void onStart() {
        super.onStart();
        if (LiveService.instance != null) {           // listening already running (e.g. returned from the pill)
            if (!liveUi) enterLiveUi();
            else {
                LiveService.listener = this::renderLive;
                renderLive();
            }
            LiveService.instance.setAppVisible(true);
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (liveUi && LiveService.instance != null) LiveService.instance.setAppVisible(false);  // show the pill
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (analyzer != null) analyzer.close();
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }
}
