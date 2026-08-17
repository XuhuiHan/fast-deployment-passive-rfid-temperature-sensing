package org.example;

import com.impinj.octane.*;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * MultiTagHeatCollector
 *
 * 多标签加热实验连续采集程序。
 *
 * 关键点：
 * 1. CSV 里的 MidDateTime / MidEpochMs / MidTime / EndDateTime / EndEpochMs / EndTime
 *    全部使用电脑系统时间 System.currentTimeMillis()。
 *
 * 2. 放电时间 persistenceTime 优先使用 reader 两次 reply 的时间差。
 *    如果 reader 时间异常，则自动退回电脑系统时间差。
 *
 * 3. 后期和温度表对齐时，优先使用 MidEpochMs / EndEpochMs。
 *
 * 输出：
 * 每个目标标签一个 CSV 文件。
 */
public class MultiTagHeatCollector {

    // =========================
    // ====== 基本配置 ==========
    // =========================

    public static final String READER_IP = "169.254.1.1";

    /**
     * 每个标签累计多少个 persistence time 后融合一次。
     */
    public static final int BURST_COUNT = 5;

    /**
     * 最大允许相邻 reply 间隔。
     * 超过这个值认为中间暂停、漏读、标签离场，不作为有效 persistence time。
     */
    public static final double MAX_INTERVAL_SEC = 10.0;

    /**
     * 最小有效间隔。
     */
    public static final double MIN_INTERVAL_SEC = 0.0;

    /**
     * 三因素融合参数。
     */
    public static final double SIGMA_RATIO = 0.015;
    public static final double RSSI_DECAY_FACTOR = 6.0;

    /**
     * 先设为 false，避免 FastID 没上报导致数据文件一直为空。
     */
    public static final boolean REQUIRE_FAST_ID = false;

    /**
     * 多标签估计数量。
     */
    public static final int TAG_POPULATION_ESTIMATE = 50;

    /**
     * 输出根目录。默认写入当前采集项目的 output/，也可通过环境变量
     * RFID_ACQUISITION_OUTPUT 指向其他实验磁盘。
     */
    public static final String OUT_DIR = resolveOutputRoot().toString();

    public static final AtomicBoolean isRunning = new AtomicBoolean(false);

    public static ImpinjReader reader;
    public static Settings settings;

    /**
     * 用户输入的目标 EPC 后四位集合。
     */
    public static final Set<String> TARGET_EPC_LAST4_SET = new LinkedHashSet<>();

    /**
     * 每个标签一个 writer。
     */
    public static final Map<String, BufferedWriter> WRITER_MAP = new LinkedHashMap<>();

    /**
     * 每个标签对应文件路径。
     */
    public static final Map<String, Path> FILE_PATH_MAP = new LinkedHashMap<>();

    public static int TARGET_TAG_COUNT = 0;

    private static final BufferedReader CONSOLE =
            new BufferedReader(new InputStreamReader(System.in));

    private static final Object writerLock = new Object();

    private static final DecimalFormatSymbols DFS =
            DecimalFormatSymbols.getInstance(Locale.US);

    static DecimalFormat dfTime = new DecimalFormat("0.0000", DFS);
    static DecimalFormat dfPhase = new DecimalFormat("0.00", DFS);
    static DecimalFormat dfRssi = new DecimalFormat("0.0", DFS);
    static DecimalFormat dfWeight = new DecimalFormat("0.00", DFS);

    private static Path resolveOutputRoot() {
        String configured = System.getenv("RFID_ACQUISITION_OUTPUT");
        if (configured != null && !configured.trim().isEmpty()) {
            return Paths.get(configured.trim()).toAbsolutePath().normalize();
        }
        return Paths.get(System.getProperty("user.dir"), "output")
                .toAbsolutePath()
                .normalize();
    }

    // =========================
    // ====== main =============
    // =========================

    public static void main(String[] args) {
        ContinuousListener listener = null;

        try {
            readTargetTagsFromConsole();

            String runName = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            Path runDir = Paths.get(OUT_DIR, runName);
            Files.createDirectories(runDir);

            openOneFilePerTag(runDir);

            System.out.println("连接读写器: " + READER_IP);

            reader = new ImpinjReader();
            reader.connect(READER_IP);

            setupReaderSettings();

            listener = new ContinuousListener();
            reader.setTagReportListener(listener);

            startKeyboardListener(listener);
            startStatusPrinter(listener);

            System.out.println();
            System.out.println("=== 程序启动 ===");
            System.out.println("目标 EPC 后四位: " + TARGET_EPC_LAST4_SET);
            System.out.println("输出目录: " + runDir);

            for (String suffix : TARGET_EPC_LAST4_SET) {
                System.out.println("标签 " + suffix + " 文件: " + FILE_PATH_MAP.get(suffix));
            }

            System.out.println(">>> 当前状态: [暂停中]");
            System.out.println(">>> 按 [回车键] 开始/暂停采集");
            System.out.println(">>> CSV 输出时间已经改为电脑系统时间 System.currentTimeMillis()");
            System.out.println("----------------------------------------------------------------");

            while (true) {
                Thread.sleep(1000);
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                isRunning.set(false);
            } catch (Exception ignored) {
            }

            try {
                if (reader != null) {
                    reader.stop();
                }
            } catch (Exception ignored) {
            }

            try {
                if (reader != null) {
                    reader.disconnect();
                }
            } catch (Exception ignored) {
            }

            closeAllWriters();
        }
    }

    // =========================
    // ====== 用户输入 ==========
    // =========================

    private static void readTargetTagsFromConsole() throws Exception {
        TARGET_EPC_LAST4_SET.clear();

        System.out.println("请输入本次实验目标标签数量：");

        while (true) {
            String line = CONSOLE.readLine();

            if (line == null) {
                throw new RuntimeException("未读取到标签数量。");
            }

            line = line.trim();

            try {
                TARGET_TAG_COUNT = Integer.parseInt(line);

                if (TARGET_TAG_COUNT <= 0) {
                    System.out.println("标签数量必须大于 0，请重新输入：");
                    continue;
                }

                break;

            } catch (Exception e) {
                System.out.println("输入错误，请输入整数，例如 3：");
            }
        }

        System.out.println("请依次输入每个目标标签 EPC 的后四位，例如 C107、A123。");

        for (int i = 0; i < TARGET_TAG_COUNT; i++) {
            while (true) {
                System.out.print("第 " + (i + 1) + " 个标签 EPC 后四位：");

                String suffix = CONSOLE.readLine();

                if (suffix == null) {
                    throw new RuntimeException("未读取到 EPC 后四位。");
                }

                suffix = suffix.trim()
                        .replace(" ", "")
                        .toUpperCase(Locale.ROOT);

                if (suffix.length() != 4) {
                    System.out.println("后四位必须正好 4 个字符，请重新输入。");
                    continue;
                }

                if (!suffix.matches("[0-9A-F]{4}")) {
                    System.out.println("后四位必须是十六进制字符 0-9/A-F，请重新输入。");
                    continue;
                }

                if (TARGET_EPC_LAST4_SET.contains(suffix)) {
                    System.out.println("这个后四位已经输入过，请重新输入。");
                    continue;
                }

                TARGET_EPC_LAST4_SET.add(suffix);
                break;
            }
        }

        System.out.println("目标 EPC 后四位集合：" + TARGET_EPC_LAST4_SET);
    }

    // =========================
    // ====== 文件管理 ==========
    // =========================

    private static void openOneFilePerTag(Path runDir) throws Exception {
        for (String suffix : TARGET_EPC_LAST4_SET) {
            Path p = runDir.resolve(suffix + ".csv");

            BufferedWriter bw = new BufferedWriter(new FileWriter(p.toFile(), true));

            bw.write("MidDateTime,MidEpochMs,MidTime,EndDateTime,EndEpochMs,EndTime,Fused_T(s),Avg_Drift(rad),Max_RSSI(dBm),Burst_Details\n");
            bw.flush();

            WRITER_MAP.put(suffix, bw);
            FILE_PATH_MAP.put(suffix, p);
        }
    }

    private static void closeAllWriters() {
        synchronized (writerLock) {
            for (BufferedWriter bw : WRITER_MAP.values()) {
                try {
                    bw.flush();
                } catch (Exception ignored) {
                }

                try {
                    bw.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    // =========================
    // ====== 键盘控制 ==========
    // =========================

    private static void startKeyboardListener(ContinuousListener listener) {
        Thread t = new Thread(() -> {
            try {
                while (true) {
                    CONSOLE.readLine();

                    boolean newState = !isRunning.get();
                    isRunning.set(newState);

                    if (newState) {
                        listener.resetRuntimeState();

                        try {
                            reader.start();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }

                        System.out.println();
                        System.out.println(">>> 指令收到: 切换为 [运行中]，reader.start()");
                    } else {
                        try {
                            reader.stop();
                        } catch (Exception ignored) {
                        }

                        listener.clearReplyStateOnly();

                        System.out.println();
                        System.out.println(">>> 指令收到: 切换为 [暂停中]，reader.stop()");
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        t.setDaemon(true);
        t.start();
    }

    private static void startStatusPrinter(ContinuousListener listener) {
        Thread t = new Thread(() -> {
            while (true) {
                try {
                    Thread.sleep(1000);

                    if (isRunning.get()) {
                        System.out.println(listener.buildStatusLine());
                    }

                } catch (Exception ignored) {
                }
            }
        });

        t.setDaemon(true);
        t.start();
    }

    // =========================
    // ====== Listener ==========
    // =========================

    static class ContinuousListener implements TagReportListener {

        private final Map<String, TagState> stateMap = new LinkedHashMap<>();

        private long reportCount = 0;
        private long allReplyCount = 0;
        private long targetReplyCount = 0;
        private long sampleCount = 0;
        private long writeCount = 0;
        private long nonTargetCount = 0;
        private long fastIdSkippedCount = 0;
        private long nullTimeCount = 0;
        private long badIntervalCount = 0;
        private long pcIntervalUsedCount = 0;
        private long readerIntervalUsedCount = 0;

        private String lastNonTargetEpc = "";
        private String lastTargetSuffix = "";
        private String lastWriteSuffix = "";

        public ContinuousListener() {
            for (String suffix : TARGET_EPC_LAST4_SET) {
                stateMap.put(suffix, new TagState());
            }
        }

        @Override
        public synchronized void onTagReported(ImpinjReader reader, TagReport report) {
            if (!MultiTagHeatCollector.isRunning.get()) {
                return;
            }

            reportCount++;

            List<Tag> tags = new ArrayList<>(report.getTags());

            /*
             * 这里只是尽量保持同一个 report 内的顺序。
             * CSV 的绝对时间不再使用 reader 时间。
             */
            tags.sort(Comparator.comparingLong(t -> {
                try {
                    if (t.getFirstSeenTime() == null) {
                        return Long.MAX_VALUE;
                    }

                    return t.getFirstSeenTime()
                            .getLocalDateTime()
                            .getTime();

                } catch (Exception e) {
                    return Long.MAX_VALUE;
                }
            }));

            for (Tag t : tags) {
                allReplyCount++;

                try {
                    handleOneTag(t);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        private void handleOneTag(Tag t) throws Exception {
            if (t.getEpc() == null) {
                return;
            }

            if (REQUIRE_FAST_ID && !t.isFastIdPresent()) {
                fastIdSkippedCount++;
                return;
            }

            String epc = t.getEpc()
                    .toString()
                    .replace(" ", "")
                    .toUpperCase(Locale.ROOT);

            String suffix = getEpcLast4(epc);

            if (!TARGET_EPC_LAST4_SET.contains(suffix)) {
                nonTargetCount++;
                lastNonTargetEpc = epc;
                return;
            }

            targetReplyCount++;
            lastTargetSuffix = suffix;

            TagState st = stateMap.computeIfAbsent(suffix, k -> new TagState());

            /*
             * 电脑系统时间：用于 CSV 绝对时间、温度对齐。
             */
            long currentPcTimeMs = System.currentTimeMillis();

            /*
             * reader 时间：只用于优先计算 persistenceTime 的间隔。
             * 如果 reader 时间为空或异常，下面会自动退回电脑系统时间差。
             */
            Long currentReaderTimeMs = safeGetReaderFirstSeenMs(t);
            if (currentReaderTimeMs == null) {
                nullTimeCount++;
            }

            double currentPhase = safeGetPhase(t);
            double currentRssi = safeGetRssi(t);

            if (st.prevPcReplyTimeMs == null) {
                st.prevPcReplyTimeMs = currentPcTimeMs;
                st.prevReaderReplyTimeMs = currentReaderTimeMs;
                st.prevPhase = currentPhase;
                st.lastEpc = epc;
                st.lastStatus = "起点";
                return;
            }

            double intervalPcSec = (currentPcTimeMs - st.prevPcReplyTimeMs) / 1000.0;

            double intervalReaderSec = Double.NaN;
            if (currentReaderTimeMs != null && st.prevReaderReplyTimeMs != null) {
                intervalReaderSec = (currentReaderTimeMs - st.prevReaderReplyTimeMs) / 1000.0;
            }

            double intervalSec;
            String intervalSource;

            if (isValidInterval(intervalReaderSec)) {
                intervalSec = intervalReaderSec;
                intervalSource = "reader";
                readerIntervalUsedCount++;
            } else if (isValidInterval(intervalPcSec)) {
                intervalSec = intervalPcSec;
                intervalSource = "pc";
                pcIntervalUsedCount++;
            } else {
                badIntervalCount++;

                st.prevPcReplyTimeMs = currentPcTimeMs;
                st.prevReaderReplyTimeMs = currentReaderTimeMs;
                st.prevPhase = currentPhase;
                st.lastEpc = epc;
                st.lastStatus = "间隔异常";
                return;
            }

            /*
             * 这一条 persistence sample 的中点时间：
             * 一律用电脑系统时间，后续温度对齐用它。
             */
            long pcMidTimeMs = (st.prevPcReplyTimeMs + currentPcTimeMs) / 2L;

            double phaseDrift = 0.0;

            if (Double.isFinite(st.prevPhase) && Double.isFinite(currentPhase)) {
                phaseDrift = calculatePhaseDiff(st.prevPhase, currentPhase);
            }

            MeasureResult res = new MeasureResult();
            res.epc = epc;
            res.epcLast4 = suffix;
            res.persistenceTime = intervalSec;
            res.phaseDrift = phaseDrift;
            res.rssi = currentRssi;
            res.prevPcReplyTimeMs = st.prevPcReplyTimeMs;
            res.currentPcReplyTimeMs = currentPcTimeMs;
            res.pcMidTimeMs = pcMidTimeMs;
            res.intervalSource = intervalSource;

            st.burst.add(res);
            st.lastEpc = epc;
            st.lastPersistence = intervalSec;
            st.lastRssi = currentRssi;
            st.lastStatus = "采样 " + st.burst.size() + "/" + BURST_COUNT + "(" + intervalSource + ")";

            sampleCount++;

            st.prevPcReplyTimeMs = currentPcTimeMs;
            st.prevReaderReplyTimeMs = currentReaderTimeMs;
            st.prevPhase = currentPhase;

            if (st.burst.size() >= BURST_COUNT) {
                List<MeasureResult> burst = new ArrayList<>(st.burst);
                st.burst.clear();

                FusionResult fused = triFactorFuse(burst);
                writeOneFusedRow(suffix, fused, burst);

                writeCount++;
                lastWriteSuffix = suffix;
                st.lastStatus = "已写入";
            }
        }

        private void writeOneFusedRow(String suffix,
                                      FusionResult fused,
                                      List<MeasureResult> burstResults) throws Exception {

            BufferedWriter bw = WRITER_MAP.get(suffix);

            if (bw == null || burstResults == null || burstResults.isEmpty()) {
                return;
            }

            /*
             * 每一轮中点时间：5 个 sample 的电脑系统中点时间取平均。
             */
            long sumMid = 0L;

            /*
             * 每一轮结束时间：5 个 sample 中最后一次 reply 到达电脑的时间。
             */
            long endTimeMs = Long.MIN_VALUE;

            for (MeasureResult r : burstResults) {
                sumMid += r.pcMidTimeMs;

                if (r.currentPcReplyTimeMs > endTimeMs) {
                    endTimeMs = r.currentPcReplyTimeMs;
                }
            }

            long fusedMidTimeMs = sumMid / Math.max(burstResults.size(), 1);

            String midDateTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
                    .format(new Date(fusedMidTimeMs));

            String midTimeOnly = new SimpleDateFormat("HH:mm:ss")
                    .format(new Date(fusedMidTimeMs));

            String endDateTime = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
                    .format(new Date(endTimeMs));

            String endTimeOnly = new SimpleDateFormat("HH:mm:ss")
                    .format(new Date(endTimeMs));

            StringBuilder details = new StringBuilder();

            for (int i = 0; i < burstResults.size(); i++) {
                MeasureResult mr = burstResults.get(i);
                double w = i < fused.debugWeights.size() ? fused.debugWeights.get(i) : 0.0;

                details.append(String.format(Locale.US,
                        "%s(RSSI=%sdBm;dphi=%s;w=%s;src=%s) ",
                        dfTime.format(mr.persistenceTime),
                        Double.isFinite(mr.rssi) ? dfRssi.format(mr.rssi) : "",
                        dfPhase.format(mr.phaseDrift),
                        dfWeight.format(w),
                        mr.intervalSource
                ));
            }

            String line = String.format(Locale.US,
                    "%s,%d,%s,%s,%d,%s,%s,%s,%s,%s\n",
                    midDateTime,
                    fusedMidTimeMs,
                    midTimeOnly,
                    endDateTime,
                    endTimeMs,
                    endTimeOnly,
                    dfTime.format(fused.fusedTime),
                    dfPhase.format(fused.avgDrift),
                    Double.isFinite(fused.maxRssi) ? dfRssi.format(fused.maxRssi) : "",
                    details.toString().trim()
            );

            synchronized (writerLock) {
                bw.write(line);
                bw.flush();
            }

            System.out.printf(Locale.US,
                    "[写入] Mid=%s End=%s 标签=%s Fused_T=%s s Avg_Drift=%s Max_RSSI=%s 文件=%s%n",
                    midTimeOnly,
                    endTimeOnly,
                    suffix,
                    dfTime.format(fused.fusedTime),
                    dfPhase.format(fused.avgDrift),
                    Double.isFinite(fused.maxRssi) ? dfRssi.format(fused.maxRssi) : "",
                    FILE_PATH_MAP.get(suffix)
            );
        }

        public synchronized void resetRuntimeState() {
            reportCount = 0;
            allReplyCount = 0;
            targetReplyCount = 0;
            sampleCount = 0;
            writeCount = 0;
            nonTargetCount = 0;
            fastIdSkippedCount = 0;
            nullTimeCount = 0;
            badIntervalCount = 0;
            pcIntervalUsedCount = 0;
            readerIntervalUsedCount = 0;

            lastNonTargetEpc = "";
            lastTargetSuffix = "";
            lastWriteSuffix = "";

            stateMap.clear();

            for (String suffix : TARGET_EPC_LAST4_SET) {
                stateMap.put(suffix, new TagState());
            }
        }

        public synchronized void clearReplyStateOnly() {
            for (TagState st : stateMap.values()) {
                st.prevPcReplyTimeMs = null;
                st.prevReaderReplyTimeMs = null;
                st.prevPhase = Double.NaN;
                st.burst.clear();
                st.lastStatus = "暂停清空";
                st.lastPersistence = Double.NaN;
                st.lastRssi = Double.NaN;
            }
        }

        public synchronized String buildStatusLine() {
            String now = new SimpleDateFormat("HH:mm:ss").format(new Date());

            StringBuilder sb = new StringBuilder();

            sb.append("[状态 ").append(now).append("] ");
            sb.append("reports=").append(reportCount);
            sb.append(" allReplies=").append(allReplyCount);
            sb.append(" targetReplies=").append(targetReplyCount);
            sb.append(" samples=").append(sampleCount);
            sb.append(" writes=").append(writeCount);
            sb.append(" badInterval=").append(badIntervalCount);
            sb.append(" intervalReader=").append(readerIntervalUsedCount);
            sb.append(" intervalPC=").append(pcIntervalUsedCount);

            if (REQUIRE_FAST_ID) {
                sb.append(" fastIdSkipped=").append(fastIdSkippedCount);
            }

            if (nullTimeCount > 0) {
                sb.append(" readerTimeNull=").append(nullTimeCount);
            }

            for (String suffix : TARGET_EPC_LAST4_SET) {
                TagState st = stateMap.get(suffix);

                sb.append(" | ").append(suffix).append(": ");

                if (st == null) {
                    sb.append("0/").append(BURST_COUNT);
                    continue;
                }

                sb.append(st.burst.size()).append("/").append(BURST_COUNT);

                if (Double.isFinite(st.lastPersistence)) {
                    sb.append(" lastT=").append(dfTime.format(st.lastPersistence)).append("s");
                }

                if (Double.isFinite(st.lastRssi)) {
                    sb.append(" RSSI=").append(dfRssi.format(st.lastRssi));
                }

                sb.append(" ").append(st.lastStatus);
            }

            if (!lastTargetSuffix.isEmpty()) {
                sb.append(" | lastTarget=").append(lastTargetSuffix);
            }

            if (!lastWriteSuffix.isEmpty()) {
                sb.append(" | lastWrite=").append(lastWriteSuffix);
            }

            if (!lastNonTargetEpc.isEmpty()) {
                sb.append(" | nonTargetLast4=").append(getEpcLast4(lastNonTargetEpc));
            }

            return sb.toString();
        }
    }

    // =========================
    // ====== 数据结构 ==========
    // =========================

    static class TagState {
        /*
         * PC 时间：用于 CSV 输出时间和温度对齐。
         */
        Long prevPcReplyTimeMs = null;

        /*
         * reader 时间：只用于优先计算放电间隔。
         * 如果为空或异常，会自动退回 PC 时间差。
         */
        Long prevReaderReplyTimeMs = null;

        double prevPhase = Double.NaN;

        List<MeasureResult> burst = new ArrayList<>();

        String lastEpc = "";
        String lastStatus = "等待";
        double lastPersistence = Double.NaN;
        double lastRssi = Double.NaN;
    }

    static class MeasureResult {
        String epc;
        String epcLast4;

        double persistenceTime;
        double phaseDrift;
        double rssi;

        /*
         * 下面三个全部是电脑系统时间。
         */
        long prevPcReplyTimeMs;
        long currentPcReplyTimeMs;
        long pcMidTimeMs;

        /*
         * reader 或 pc，表示这条 persistenceTime 用哪个时间差算出来。
         */
        String intervalSource = "";
    }

    static class FusionResult {
        double fusedTime;
        double avgDrift;
        double maxRssi;
        List<Double> debugWeights = new ArrayList<>();
    }

    // =========================
    // ====== 融合函数 ==========
    // =========================

    private static FusionResult triFactorFuse(List<MeasureResult> rawList) {
        FusionResult result = new FusionResult();

        if (rawList == null || rawList.isEmpty()) {
            result.fusedTime = Double.NaN;
            result.avgDrift = Double.NaN;
            result.maxRssi = Double.NaN;
            return result;
        }

        List<Double> times = new ArrayList<>();

        double maxRssiInBurst = Double.NEGATIVE_INFINITY;

        for (MeasureResult r : rawList) {
            times.add(r.persistenceTime);

            if (Double.isFinite(r.rssi) && r.rssi > maxRssiInBurst) {
                maxRssiInBurst = r.rssi;
            }
        }

        if (!Double.isFinite(maxRssiInBurst)) {
            maxRssiInBurst = Double.NaN;
        }

        Collections.sort(times);

        double median = times.get(times.size() / 2);
        double dynamicSigma = Math.max(0.005, median * SIGMA_RATIO);

        double weightedSum = 0.0;
        double totalWeight = 0.0;
        double driftSum = 0.0;

        result.maxRssi = maxRssiInBurst;

        for (MeasureResult res : rawList) {
            double timeDiff = Math.abs(res.persistenceTime - median);
            double wTime = Math.exp(-timeDiff / dynamicSigma);

            double wPhase = Math.exp(-Math.abs(res.phaseDrift));

            double wRssi = 1.0;

            if (Double.isFinite(maxRssiInBurst) && Double.isFinite(res.rssi)) {
                double rssiDiff = maxRssiInBurst - res.rssi;
                wRssi = Math.exp(-rssiDiff / RSSI_DECAY_FACTOR);
            }

            double finalWeight = wTime * wPhase * wRssi;

            if (!Double.isFinite(finalWeight)) {
                finalWeight = 0.0;
            }

            result.debugWeights.add(finalWeight);

            weightedSum += res.persistenceTime * finalWeight;
            totalWeight += finalWeight;
            driftSum += res.phaseDrift;
        }

        if (totalWeight > 0) {
            result.fusedTime = weightedSum / totalWeight;
        } else {
            result.fusedTime = median;
        }

        result.avgDrift = driftSum / rawList.size();

        return result;
    }

    // =========================
    // ====== 相位差 ============
    // =========================

    private static double calculatePhaseDiff(double phase1, double phase2) {
        double diff = phase2 - phase1;

        while (diff > Math.PI) {
            diff -= 2.0 * Math.PI;
        }

        while (diff < -Math.PI) {
            diff += 2.0 * Math.PI;
        }

        return Math.abs(diff);
    }

    // =========================
    // ====== Reader 设置 =======
    // =========================

    private static void setupReaderSettings() throws Exception {
        settings = reader.queryDefaultSettings();

        settings.setReaderMode(ReaderMode.MaxThroughput);
        settings.setSearchMode(SearchMode.SingleTarget);
        settings.setSession(1);
        settings.setTagPopulationEstimate(Math.max(TAG_POPULATION_ESTIMATE, TARGET_TAG_COUNT));

        List<AntennaConfig> antennas =
                settings.getAntennas().getAntennaConfigs();

        for (AntennaConfig ac : antennas) {
            ac.setEnabled(false);
            ac.setTxPowerinDbm(32.0);
            ac.setIsMaxRxSensitivity(true);
        }

        if (!antennas.isEmpty()) {
            antennas.get(0).setEnabled(true);
        }

        ReportConfig r = settings.getReport();

        r.setIncludeFirstSeenTime(true);
        r.setIncludeLastSeenTime(true);
        r.setIncludeFastId(true);
        r.setIncludePhaseAngle(true);
        r.setIncludePeakRssi(true);
        r.setMode(ReportMode.Individual);

        settings.setReport(r);

        /*
         * 不使用 reader 端 EPC filter。
         * 多个目标标签由程序端按 EPC 后四位筛选。
         */
        reader.applySettings(settings);

        System.out.println("Reader 设置完成。");
        System.out.println("程序端目标 EPC 后四位: " + TARGET_EPC_LAST4_SET);
        System.out.println("REQUIRE_FAST_ID = " + REQUIRE_FAST_ID);
    }

    // =========================
    // ====== 安全读取 ==========
    // =========================

    private static Long safeGetReaderFirstSeenMs(Tag t) {
        try {
            if (t.getFirstSeenTime() == null) {
                return null;
            }

            return t.getFirstSeenTime()
                    .getLocalDateTime()
                    .getTime();

        } catch (Exception e) {
            return null;
        }
    }

    private static double safeGetPhase(Tag t) {
        try {
            return t.getPhaseAngleInRadians();
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private static double safeGetRssi(Tag t) {
        try {
            return t.getPeakRssiInDbm();
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    private static boolean isValidInterval(double intervalSec) {
        return Double.isFinite(intervalSec)
                && intervalSec > MIN_INTERVAL_SEC
                && intervalSec <= MAX_INTERVAL_SEC;
    }

    private static String getEpcLast4(String epc) {
        if (epc == null) {
            return "";
        }

        String clean = epc.replace(" ", "").toUpperCase(Locale.ROOT);

        if (clean.length() < 4) {
            return clean;
        }

        return clean.substring(clean.length() - 4);
    }
}
