#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QDateTime>
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QFileInfo>
#include <QDir>
#include <QResizeEvent>
#include <QtConcurrent/QtConcurrent>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════════════════════════

// Model
static const std::string ONNX_PATH    = "D:/Sehfeld/model/sehfeld_backbone.onnx";
static const std::string MEMBANK_PATH = "D:/Sehfeld/model/sehfeld_memory_bank.bin";
static const float       THRESHOLD    = 4.2545f;

// Colors
static const char* C_BG      = "#0a0e1a";
static const char* C_SURFACE = "#0d1220";
static const char* C_BORDER  = "#1e2640";
static const char* C_ACCENT  = "#4fc3f7";
static const char* C_TEXT    = "#8ba7cc";
static const char* C_DIM     = "#2a3550";
static const char* C_GREEN   = "#00e676";
static const char* C_RED     = "#ff5252";
static const char* C_AMBER   = "#ffab40";

// Fonts
static const char* FONT_UI   = "Segoe UI";
static const char* FONT_MONO = "Consolas";

// ═══════════════════════════════════════════════════════════════════════════
//  CONSTRUCTOR
// ═══════════════════════════════════════════════════════════════════════════

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    showMaximized();

    // ── DPI scale ─────────────────────────────────────────────────────────────
    sc_          = qMax(QGuiApplication::primaryScreen()->logicalDotsPerInch() / 96.0f, 1.0f);
    H_HEADER_    = qRound(46  * sc_);
    W_LEFT_      = qRound(240 * sc_);
    H_STRIP_     = qRound(118 * sc_);
    H_FEED_HDR_  = qRound(26  * sc_);
    PAD_TAB_     = qRound(18  * sc_);
    PAD_CONTENT_ = qRound(8   * sc_);
    FONT_BASE_   = qRound(12  * sc_);
    FONT_SMALL_  = qRound(9   * sc_);
    FONT_MONO_   = qRound(11  * sc_);

    // ── Left panel margins ────────────────────────────────────────────────────
    ui->wLeft->layout()->setContentsMargins(0, 0, 0, 0);
    ui->wLeft->layout()->setSpacing(0);
    if (auto *l = ui->wStats->layout())
        l->setContentsMargins(PAD_CONTENT_, PAD_CONTENT_, PAD_CONTENT_, PAD_CONTENT_);

    setupStyles();
    setupDetector();
    setupCategories();

    // Force top row border-bottom
    ui->wTopRow->setAutoFillBackground(false);
    ui->wTopRow->setStyleSheet(
        "background: transparent; border-top: 1px solid " + QString(C_BORDER) + ";"
    );
    ui->lblSecModel->setStyleSheet(QString(
        "background: %1; border-left: 1px solid %2; border-top: 1px solid %2; border-right: 1px solid %2; border-bottom: 1px solid %2; "
        "font-family: '%3'; font-size: %4px; letter-spacing: 2px; color: %5; padding: 0 14px 0 8px;"
    ).arg(C_SURFACE, C_BORDER, FONT_MONO, QString::number(FONT_BASE_), C_ACCENT));

    // Clock
    clockTimer_ = new QTimer(this);
    connect(clockTimer_, &QTimer::timeout, this, &MainWindow::updateClock);
    clockTimer_->start(1000);
    updateClock();

    // Online + Clock: same horizontal padding as tabs
    ui->lblOnlineWord->setContentsMargins(PAD_TAB_, 0, PAD_TAB_, 0);
    ui->lblClock->setContentsMargins(PAD_TAB_, 0, PAD_TAB_, 0);

    // Force feed backgrounds
    auto feedPalette = ui->lblInputImage->palette();
    feedPalette.setColor(QPalette::Window, QColor(C_BG));
    feedPalette.setColor(QPalette::WindowText, QColor(C_TEXT));
    for (auto *w : QList<QWidget*>{ui->lblInputImage, ui->lblHeatmap, ui->wFeedInput, ui->wFeedHeatmap}) {
        w->setAutoFillBackground(true);
        w->setPalette(feedPalette);
    }
    // Force feed backgrounds via parent
    ui->wFeedInput->setStyleSheet(QString("QWidget { background: %1; } QWidget#wFeedInput { background: %1; border: 1px solid %2; }").arg(C_BG, C_BORDER));
    ui->wFeedHeatmap->setStyleSheet(QString("QWidget { background: %1; } QWidget#wFeedHeatmap { background: %1; border: 1px solid %2; }").arg(C_BG, C_BORDER));
    ui->lblInputMeta->setVisible(false);

    // Connections
    connect(ui->btnTabLive,      &QPushButton::clicked, this, &MainWindow::onTabLive);
    connect(ui->btnTabBatch,     &QPushButton::clicked, this, &MainWindow::onTabBatch);
    connect(ui->btnTabHistory,   &QPushButton::clicked, this, &MainWindow::onTabHistory);
    connect(ui->btnLoad,         &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(ui->btnRun,          &QPushButton::clicked, this, &MainWindow::onRunDetection);
    connect(ui->btnExport,       &QPushButton::clicked, this, &MainWindow::onExportReport);
    connect(ui->btnBatchLoad,    &QPushButton::clicked, this, &MainWindow::onBatchLoad);
    connect(ui->btnBatchRun,     &QPushButton::clicked, this, &MainWindow::onBatchRun);
    connect(ui->btnBatchExport,  &QPushButton::clicked, this, &MainWindow::onBatchExport);
    connect(ui->btnExportAll,    &QPushButton::clicked, this, &MainWindow::onExportAll);
    connect(ui->btnClearHistory, &QPushButton::clicked, this, &MainWindow::onClearHistory);

    watcher_ = new QFutureWatcher<DetectionResult>(this);
    connect(watcher_, &QFutureWatcher<DetectionResult>::finished,
            this,     &MainWindow::onDetectionFinished);

    // History table columns
    ui->tblHistory->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);
    ui->tblHistory->verticalHeader()->setVisible(false);
    ui->tblHistory->setColumnWidth(0,  42);
    ui->tblHistory->setColumnWidth(1, 148);
    ui->tblHistory->setColumnWidth(3,  68);
    ui->tblHistory->setColumnWidth(4,  58);
    ui->tblHistory->setColumnWidth(5,  98);
    ui->tblHistory->setColumnWidth(6,  72);

    ui->btnClearHistory->setEnabled(false);
    ui->lblBatchSummary->setText(" ");
    setTabActive(0);
    updateSessionStats();
    log("SEHFELD v1.0.0 — System ready.");
}

MainWindow::~MainWindow()
{
    delete detector_;
    delete ui;
}

// ═══════════════════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════════════════

void MainWindow::setupDetector()
{
    log("Loading model...");
    detector_ = new AnomalyDetector(ONNX_PATH, MEMBANK_PATH, THRESHOLD);
    if (detector_->isReady()) {
        log("Model ready.  Memory bank: 21401 patches.");
        statusBar()->showMessage(
            "System online  ·  PatchCore / WideResNet-50  ·  MVTec AD Bottle  ·  thr 4.2545", 0);
    } else {
        log("ERROR: Model failed to load. Check paths in mainwindow.cpp.");
        ui->lblOnlineWord->setStyleSheet(
            QString("color:%1; font-family:%2; font-size:11px; font-weight:600;")
            .arg(C_RED, FONT_MONO));
        ui->lblOnlineWord->setText("Offline");
        ui->btnRun->setEnabled(false);
    }
}

void MainWindow::setupCategories()
{
    ui->lblModelName->setWordWrap(true);
}

void MainWindow::setupStyles()
{
    QFont f(FONT_UI, FONT_BASE_);
    qApp->setFont(f);

    // NOTE: To change any visual property, edit the CONFIG section at the top.
    // The %N placeholders map to CONFIG constants — do not change the order.
    setStyleSheet(QString(R"(

/* ── Base ── */
QMainWindow, QWidget {
    background: %1; color: %5;
    font-family: "%10"; font-size: %11px;
}

/* ── Header ── */
#wHeader { background: %1;}

#lblBrand {
    font-family: "%12"; font-size: 20px; font-weight: bold;
    color: %4; letter-spacing: 2px;
    padding: 0 40px 0 18px;
    border-right: 1px solid %3;
    border-right: 1px solid %3;
    border-top: 1px solid %3;
    border-left: 1px solid %3;
    min-height: %13px; max-height: %13px;
}
#btnTabLive, #btnTabBatch, #btnTabHistory {
    background: transparent; border: none;
    border-bottom: 2px solid transparent; border-radius: 0;
    color: %6; font-size: 11px; font-weight: 500;
    padding: 0 %14px; min-height: %13px; max-height: %13px;
}
#btnTabLive:hover, #btnTabBatch:hover, #btnTabHistory:hover { color: %5; }
#btnTabLive:checked, #btnTabBatch:checked, #btnTabHistory:checked {
    color: %4; border-bottom: 2px solid %4;
}
#lblOnlineWord {
    color: %7; font-family: "%12"; font-size: 11px; font-weight: 600;
    background: transparent; min-height: %13px; max-height: %13px;
}
#lblClock {
    color: %4; font-family: "%12"; font-size: 11px;
    background: transparent; min-height: %13px; max-height: %13px;
}

/* ── Left panel ── */
#wLeft { background: 1; border-right: 1px solid %3; border-left: 1px solid %3; }

#wTopRow { background: %2; }
#lblSecCategory, #lblSecStats, #lblSecLog {
    font-family: "%12"; font-size: %17px; letter-spacing: 2px; color: %4;
    padding: 0 14px 0 %15px;
    min-height: 38px; max-height: 38px;
    background: %2; border: 1px solid %3;
}
#wMcard {
    background: %1;
    border: 1px solid %3; border-radius: 0px;
    margin: 4px 6px;
}
#lstCategories {
    background: %1; border: 1px solid %3;
    font-family: "%12"; font-size: %18px;
}
#lstCategories::item { padding: 5px %15px; border-bottom: 1px solid %3; }
#lstCategories::item:selected {
    background: rgba(79,195,247,0.08); color: %4; border-left: 2px solid %4;
}

#lblModelName { color: %4; font-size: 16px; font-weight: bold; qproperty-alignment: AlignCenter; }
#lblModelSub  { color: #2e6a9e; font-family: "%12"; font-size: 14px; qproperty-alignment: AlignCenter; }

#wKpi1, #wKpi2, #wKpi3, #wKpi4 {
    background: %1; border: 1px solid %3;
    border-radius: 0px; padding: 6px %15px;
}
#wKpi2, #wKpi4 { }
#lblKv1, #lblKv2, #lblKv3, #lblKv4 {
    color: %4; font-family: "%12"; font-size: %17px; font-weight: bold;
}
#lblKl1, #lblKl2, #lblKl3, #lblKl4 { color: %4; font-size: %17px; font-family: "%12"; }

#wStats { background: %1; border: 1px solid %3; margin: 4px 6px; }
#lblStInspK, #lblStPassK, #lblStDefK, #lblStRateK { color: %4; font-size: %17px; font-family: "%12"; }
#lblStInspV, #lblStPassV, #lblStDefV, #lblStRateV {
    color: %4; font-family: "%12"; font-size: %17px; font-weight: bold;
}

#txtLog {
    background: %1; border: 1px solid %3; margin: 4px 6px;
    color: %4; font-family: "%12"; font-size: %17px;
}

/* ── Toolbar ── */
#wTopRow { background: %1; border-bottom: 1px solid %3; }
#wToolbar {
    border-right: 1px solid %3;
    background: %1;
    border-radius: 0;
}

#wBatchToolbar, #wHistToolbar {
    background: %2;
    border-top: 1px solid %3;
    border-bottom: 1px solid %3;
    border-left: 1px solid %3;
    border-right: 1px solid %3;
    border-radius: 0;
}
#lblFileInfo { font-family: "%12"; font-size: %17px; color: %4; padding: 0 10px; }

/* ── Buttons ── */
QPushButton {
    background: transparent; border: none;
    border-bottom: 2px solid transparent; border-radius: 0;
    color: %6; font-size: %18px; font-weight: 500;
    padding: 0 12px; min-height: 38px; max-height: 38px;
}
QPushButton:hover   { color: %5; }
QPushButton:pressed { color: %4; }
QPushButton:disabled { color: %6; }

#btnLoad, #btnRun, #btnExport,
#btnBatchLoad, #btnBatchRun, #btnBatchExport,
#btnExportAll {
    background: transparent; border: none;
    border-bottom: 2px solid transparent; border-radius: 0;
    color: %6; font-weight: 500;
    min-height: 38px; max-height: 38px;
}
#btnLoad:enabled, #btnRun:enabled, #btnExport:enabled,
#btnBatchLoad:enabled, #btnBatchRun:enabled, #btnBatchExport:enabled,
#btnExportAll:enabled {
    color: %4; border-bottom: 2px solid %4;
}
#btnLoad:disabled, #btnRun:disabled, #btnExport:disabled,
#btnBatchLoad:disabled, #btnBatchRun:disabled, #btnBatchExport:disabled,
#btnExportAll:disabled {
    color: %6; border-bottom: 2px solid transparent;
}
#btnLoad:hover:enabled, #btnRun:hover:enabled, #btnExport:hover:enabled,
#btnBatchLoad:hover:enabled, #btnBatchRun:hover:enabled, #btnBatchExport:hover:enabled,
#btnExportAll:hover:enabled { color: %5; }
#btnClearHistory { color: %6; border: none; border-radius: 0; }
#btnClearHistory:enabled { color: %8; }
#btnClearHistory:hover:enabled { background: rgba(255,82,82,0.08); }

/* ── Feed panels ── */
#wFeedInput, #wFeedHeatmap { background: %1; border-top: 1px solid %3; }
#wFiHdr, #wFhHdr { background: %2; border-right: 1px solid %3; }
#lblInputTitle, #lblHeatmapTitle {
    font-family: "%12"; font-size: %16px; letter-spacing: 1.5px; color: %4;
}
#lblInputMeta {
    font-family: "%12"; font-size: %16px; color: %4;
    padding: 0 6px; margin-left: 4px;
}
#lblInputImage, #lblHeatmap { background: %2; color: %5; font-size: 11px; border-right: 1px solid %3; border-top: 1px solid %3; border-bottom: 1px solid %3; }

/* ── Result strip ── */
#wResultStrip { background: %1; border-top: 1px solid %3; border-right: 1px solid %3; }
#wSVerdict { background: %1; border-right: 1px solid %3; padding: 0 20px; }
#lblSVIcon  { font-size: %17px; font-weight: 600; font-family: "%12"; color: %4; background: transparent; }
#lblSVScore { font-size: %17px; font-weight: bold; font-family: "%12"; color: %4; background: transparent; }
#lblSVSub   { font-family: "%12"; font-size: %17px; color: %4; background: transparent; }
#wSBar { border-right: 1px solid %3;}
#lblSBLbl { font-size: %17px; font-family: "%12"; color: %4; font-weight: 500; }
#lblSBVal { font-family: "%12"; font-size: %17px; font-weight: bold; color: %4; }
#wSDetails { padding: 0 20px; border-right: 1px solid %3; }
#lblSDLocK { font-size: %17px; font-family: "%12"; color: %4; }
#lblSDLocV { font-family: "%12"; font-size: %17px; color: %4;}

/* ── Score bars ── */
QProgressBar#scoreBar { background: %3; border: none; border-radius: 0; }
QProgressBar#scoreBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 %7,stop:0.5 %9,stop:1 %8);
    border-radius: 0;
}
QProgressBar#progressBar, QProgressBar#batchProgress {
    background: %3; border: none; border-radius: 0;
}
QProgressBar#progressBar::chunk, QProgressBar#batchProgress::chunk {
    background: %4; border-radius: 0;
}

/* ── Batch ── */
#lstBatch {
    background: %1; alternate-background-color: %2; border: none;
    color: %4; font-family: "%12"; font-size: %17px;
}
#lstBatch::item { padding: 8px 14px; border-bottom: 1px solid %3; }
#lstBatch::item:selected { background: rgba(79,195,247,0.1); color: %4; }
#lblBatchStatus { background: %2; border-top: 1px solid %3; border-bottom: 1px solid %3; font-family: "%12"; font-size: %17px; color: %4; padding: 0 12px; }
#lblBatchSummary {
    background: %2; border-top: 1px solid %3; border-bottom: 1px solid %3;
    color: %4; font-family: "%12"; font-size: %17px; padding: 0 14px;
}

/* ── History ── */
#lblHistCount {
    background: %2; border-top: 1px solid %3; border-bottom: 1px solid %3;
    font-family: "%12"; font-size: %17px; color: %4; padding: 0 12px;
}
QTableWidget {
    background: %1; alternate-background-color: %2; border: none;
    color: %5; gridline-color: %3; font-family: "%12"; font-size: 11px;
    selection-background-color: rgba(79,195,247,0.1); selection-color: %4; outline: none;
}
QTableWidget::item { padding: 7px 8px; border-bottom: 1px solid %3; }
QHeaderView::section {
    background: %2; border: none;
    border-bottom: 1px solid %3; border-right: 1px solid %3;
    padding: 7px 8px; font-family: "%12"; font-size: 9px;
    font-weight: bold; color: %4; letter-spacing: 1px;
}
QHeaderView::section:last { border-right: none; }

/* ── Scrollbars ── */
QScrollBar:vertical   { background: %1; width: 5px; border: none; }
QScrollBar:horizontal { background: %1; height: 5px; border: none; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: %3; border-radius: 0; min-height: 20px; min-width: 20px;
}
QScrollBar::handle:vertical:hover,
QScrollBar::handle:horizontal:hover { background: %6; }
QScrollBar::add-line:vertical,  QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { height: 0; width: 0; }

/* ── Dividers ── */
QFrame[frameShape="5"] { color: %3; max-width: 1px; min-width: 1px; }

/* ── Status bar ── */
QStatusBar {
    background: %2; border-top: 1px solid %3;
    color: %4; font-family: "%12"; font-size: %17px; padding: 0 8px;
}

    )")
    .arg(C_BG,                          // %1
         C_SURFACE,                     // %2
         C_BORDER,                      // %3
         C_ACCENT,                      // %4
         C_TEXT,                        // %5
         C_DIM,                         // %6
         C_GREEN,                       // %7
         C_RED,                         // %8
         C_AMBER)                       // %9
    .arg(FONT_UI,                       // %10
         QString::number(FONT_BASE_),   // %11
         FONT_MONO,                     // %12
         QString::number(H_HEADER_),    // %13
         QString::number(PAD_TAB_),     // %14
         QString::number(PAD_CONTENT_), // %15
         QString::number(FONT_SMALL_),  // %16  small text
         QString::number(FONT_BASE_),   // %17  normal/bold text
         QString::number(qRound(10 * sc_)) // %18 medium text
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════════

void MainWindow::updateClock()
{
    ui->lblClock->setText(QDateTime::currentDateTime().toString("hh:mm:ss"));
}

void MainWindow::log(const QString &msg)
{
    ui->txtLog->append(
        QString("[%1]  %2").arg(QDateTime::currentDateTime().toString("hh:mm:ss"), msg));
}

QPixmap MainWindow::matToPixmap(const cv::Mat &img) const
{
    if (img.empty()) return {};
    cv::Mat rgb;
    if      (img.channels() == 3) cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    else if (img.channels() == 1) cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    else                          rgb = img.clone();
    QImage qi(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(qi.copy());
}

void MainWindow::displayImage(QLabel *lbl, const cv::Mat &img)
{
    if (img.empty()) return;
    lbl->setPixmap(
        matToPixmap(img).scaled(lbl->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}



void MainWindow::setTabActive(int idx)
{
    ui->btnTabLive->setChecked(idx == 0);
    ui->btnTabBatch->setChecked(idx == 1);
    ui->btnTabHistory->setChecked(idx == 2);
    ui->stackedWidget->setCurrentIndex(idx);
}

void MainWindow::setStripState(bool isDefect)
{
    const char* bc = isDefect ? C_RED   : C_GREEN;
    ui->wResultStrip->setStyleSheet(QString(
        "#wResultStrip { background:%1; border-top:1px solid %4; border-right:1px solid %4; }"
        "#wSVerdict    { background:%1; border-right:1px solid %4; padding:0 20px; }"
        "#lblSVIcon    { font-size:14px; font-weight:600; color:%2; }"
        "#lblSVScore   { font-size:14px; font-weight:bold; font-family:Consolas; color:%2; }"
        "#lblSBVal     { font-family:Consolas; font-size:14px; font-weight:bold; color:%2; }"
    ).arg(C_SURFACE, bc, C_SURFACE, C_BORDER,
         QString::number(FONT_BASE_),
         QString::number(FONT_BASE_ + 4),
         QString::number(FONT_MONO_)));
}

void MainWindow::updateSessionStats()
{
    ui->lblStInspV->setText(QString::number(sampleCount_));
    ui->lblStPassV->setText(QString::number(passCount_));
    ui->lblStDefV->setText(QString::number(defectCount_));
    if (sampleCount_ > 0) {
        float rate = (float)defectCount_ / sampleCount_ * 100.0f;
        ui->lblStRateV->setText(QString("%1%").arg(rate, 0, 'f', 1));
        ui->lblStDefV->setStyleSheet(
            defectCount_ > 0
            ? QString("font-family:Consolas;font-size:%1px;font-weight:bold;color:%2;").arg(FONT_MONO_).arg(C_RED)
            : QString("font-family:Consolas;font-size:%1px;font-weight:bold;color:%2;").arg(FONT_MONO_).arg(C_GREEN));
    } else {
        ui->lblStRateV->setText("—");
    }
}

void MainWindow::resizeEvent(QResizeEvent *ev)
{
    QMainWindow::resizeEvent(ev);
    if (!currentImage_.empty())
        displayImage(ui->lblInputImage, currentImage_);
    if (!lastResult_.heatmap.empty() && !currentImage_.empty()) {
        cv::Mat blend, hr;
        cv::resize(lastResult_.heatmap, hr, currentImage_.size());
        cv::addWeighted(currentImage_, 0.35, hr, 0.65, 0, blend);
        displayImage(ui->lblHeatmap, blend);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  TABS
// ═══════════════════════════════════════════════════════════════════════════

void MainWindow::onTabLive()    { setTabActive(0); }
void MainWindow::onTabBatch()   { setTabActive(1); }
void MainWindow::onTabHistory() { setTabActive(2); }

// ═══════════════════════════════════════════════════════════════════════════
//  LIVE
// ═══════════════════════════════════════════════════════════════════════════

void MainWindow::onLoadImage()
{
    QString path = QFileDialog::getOpenFileName(
        this, "Open Image", "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)");
    if (path.isEmpty()) return;

    currentImagePath_ = path;
    currentImage_     = cv::imread(path.toStdString());
    if (currentImage_.empty()) {
        QMessageBox::warning(this, "Open Image", "Could not open:\n" + path);
        return;
    }

    displayImage(ui->lblInputImage, currentImage_);

    QString fname = QFileInfo(path).fileName();
    ui->lblInputMeta->setText(
        QString("%1  ·  %2×%3").arg(fname).arg(currentImage_.cols).arg(currentImage_.rows));
    ui->lblInputMeta->setVisible(true);
    ui->lblFileInfo->setText("");

    ui->lblHeatmap->clear();
    ui->lblHeatmap->setText("Run inspection to see heatmap");
    ui->lblSVIcon->setText("—");
    ui->lblSVSub->setText(QString("Sample #%1").arg(sampleCount_ + 1));
    ui->lblSBVal->setText("—");
    ui->lblSDLocV->setText("—");
    ui->scoreBar->setValue(0);
    ui->wResultStrip->setStyleSheet("");

    ui->btnRun->setEnabled(detector_ && detector_->isReady());
    ui->btnExport->setEnabled(false);

    log(QString("Opened: %1  [%2×%3]").arg(fname).arg(currentImage_.cols).arg(currentImage_.rows));
    statusBar()->showMessage(
        QString("Image loaded  ·  %1  ·  %2×%3 px")
        .arg(fname).arg(currentImage_.cols).arg(currentImage_.rows), 0);
}

void MainWindow::onRunDetection()
{
    if (currentImage_.empty() || !detector_ || !detector_->isReady()) return;

    ui->btnLoad->setEnabled(false);
    ui->btnRun->setEnabled(false);
    ui->btnExport->setEnabled(false);
    ui->progressBar->setVisible(true);
    ui->lblSVIcon->setText("Analyzing...");

    log("Running inspection...");
    elapsedTimer_.start();

    cv::Mat img = currentImage_.clone();
    std::string p = currentImagePath_.toStdString();
    AnomalyDetector *det = detector_;

    watcher_->setFuture(QtConcurrent::run([det, img, p]() {
        return det->inspect(img, p);
    }));
}

void MainWindow::onDetectionFinished()
{
    qint64 ms   = elapsedTimer_.elapsed();
    lastResult_ = watcher_->result();

    sampleCount_++;
    if (lastResult_.is_defect) defectCount_++; else passCount_++;
    updateSessionStats();

    showResult(lastResult_, ms);
    addHistory(lastResult_, ms);

    ui->btnLoad->setEnabled(true);
    ui->btnRun->setEnabled(false);
    ui->progressBar->setVisible(false);
    ui->btnExport->setEnabled(true);

    QString v = lastResult_.is_defect ? "DEFECT" : "PASS";
    log(QString("Done  ·  %1  ·  score=%2  ·  %3 ms")
        .arg(v).arg(lastResult_.anomaly_score, 0, 'f', 2).arg(ms));
    statusBar()->showMessage(
        QString("Sample #%1  ·  %2  ·  score=%3  ·  %4 ms")
        .arg(sampleCount_).arg(v)
        .arg(lastResult_.anomaly_score, 0, 'f', 2).arg(ms), 0);
}

void MainWindow::showResult(const DetectionResult &r, qint64 ms)
{
    if (!r.heatmap.empty()) {
        cv::Mat blend, hr;
        cv::resize(r.heatmap, hr, currentImage_.size());
        cv::addWeighted(currentImage_, 0.35, hr, 0.65, 0, blend);
        displayImage(ui->lblHeatmap, blend);
    }

    float maxE = r.threshold * 3.0f;
    ui->scoreBar->setValue((int)(std::min(r.anomaly_score / maxE, 1.0f) * 1000.0f));
    setStripState(r.is_defect);

    ui->lblSVIcon->setText(r.is_defect ? "⚠  DEFECT DETECTED" : "✔  PASS");
    ui->lblSVSub->setText(
        QString("Sample #%1  ·  Frame #%2").arg(sampleCount_).arg(r.frame_id));
    ui->lblSBVal->setText(
        QString("%1  /  %2").arg(r.anomaly_score, 0, 'f', 2).arg(r.threshold, 0, 'f', 2));

    ui->lblSDLocV->setText(
        QString("X %1    Y %2").arg(r.defect_location.x).arg(r.defect_location.y));
    ui->lblKv2->setText(QString("%1 ms").arg(ms));
}

void MainWindow::onExportReport()
{
    QString date     = QDateTime::currentDateTime().toString("yyMMdd");
    QString imgName  = QFileInfo(currentImagePath_).fileName();
    QString defName  = QString("%1_sehfeld_img_%2.json").arg(date, imgName);
    QString path = QFileDialog::getSaveFileName(
        this, "Export Report", defName, "JSON (*.json)");
    if (path.isEmpty()) return;
    if (Reporter::exportJSON(lastResult_, path.toStdString())) {
        log("Exported: " + QFileInfo(path).fileName());
        statusBar()->showMessage("Saved: " + path, 5000);
    } else {
        QMessageBox::critical(this, "Export Failed", "Cannot write:\n" + path);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  BATCH
// ═══════════════════════════════════════════════════════════════════════════

void MainWindow::onBatchLoad()
{
    QString dir = QFileDialog::getExistingDirectory(this, "Open Folder");
    if (dir.isEmpty()) return;

    QDir qdir(dir);
    QStringList files = qdir.entryList(
        {"*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"}, QDir::Files, QDir::Name);

    batchFiles_.clear();
    ui->lstBatch->clear();
    for (const QString &f : files) {
        batchFiles_ << qdir.filePath(f);
        ui->lstBatch->addItem("  " + f);
    }

    batchFolderName_ = QFileInfo(dir).fileName();
    ui->lblBatchStatus->setText(
        QString("%1 images  ·  %2").arg(batchFiles_.size()).arg(batchFolderName_));
    ui->btnBatchRun->setEnabled(!batchFiles_.isEmpty() && detector_ && detector_->isReady());
    ui->btnBatchExport->setEnabled(false);
    log(QString("Batch: %1 images").arg(batchFiles_.size()));
}

void MainWindow::onBatchRun()
{
    if (batchFiles_.isEmpty() || !detector_ || !detector_->isReady()) return;

    ui->btnBatchLoad->setEnabled(false);
    ui->btnBatchRun->setEnabled(false);
    ui->btnClearHistory->setEnabled(false);
    ui->btnBatchExport->setEnabled(false);
    ui->batchProgress->setVisible(true);
    ui->batchProgress->setMaximum(batchFiles_.size());
    ui->batchProgress->setValue(0);
    ui->lblBatchSummary->setText("  Running...");

    QStringList files = batchFiles_;
    AnomalyDetector *det = detector_;

    auto future = QtConcurrent::run([this, det, files]() {
        std::vector<DetectionResult> results;
        int i = 0;
        for (const QString &fp : files) {
            QElapsedTimer t; t.start();
            cv::Mat img = cv::imread(fp.toStdString());
            if (!img.empty()) {
                DetectionResult r = det->inspect(img, fp.toStdString());
                qint64 ms = t.elapsed();
                results.push_back(r);
                QMetaObject::invokeMethod(this, [this, r, i, fp, ms]() {
                    ui->batchProgress->setValue(i + 1);
                    QString v   = r.is_defect ? "DEFECT" : "PASS  ";
                    QString txt = QString("  %1    %2  %3    %4 ms")
                        .arg(QFileInfo(fp).fileName(), -30)
                        .arg(v).arg(r.anomaly_score, 6, 'f', 2).arg(ms, 4);
                    if (ui->lstBatch->item(i)) {
                        ui->lstBatch->item(i)->setText(txt);
                        ui->lstBatch->item(i)->setForeground(
                            r.is_defect ? QColor(C_RED) : QColor(C_GREEN));
                    }
                    addHistory(r, ms);
                }, Qt::QueuedConnection);
            }
            i++;
        }
        QMetaObject::invokeMethod(this, [this, results]() {
            int def = 0;
            for (const auto &r : results) {
                if (r.is_defect) def++;
            }
            ui->lblBatchSummary->setText(
                QString("  Complete  ·  %1 total  ·  %2 defects  ·  %3 passed")
                .arg(results.size()).arg(def).arg((int)results.size() - def));
            ui->btnBatchLoad->setEnabled(true);
            ui->btnBatchRun->setEnabled(true);
            ui->btnClearHistory->setEnabled(!history_.isEmpty());
            ui->btnBatchExport->setEnabled(true);
            ui->batchProgress->setVisible(false);
            log(QString("Batch done  ·  %1 defects / %2").arg(def).arg(results.size()));
        }, Qt::QueuedConnection);
    });
    Q_UNUSED(future);
}

void MainWindow::onBatchExport()
{
    if (history_.isEmpty()) {
        QMessageBox::information(this, "Export All", "No records to export.");
        return;
    }
    QString date     = QDateTime::currentDateTime().toString("yyMMdd");
    QString defName  = QString("%1_sehfeld_batch_%2.json").arg(date, batchFolderName_);
    QString path = QFileDialog::getSaveFileName(
        this, "Export Batch Report", defName, "JSON (*.json)");
    if (path.isEmpty()) return;

    std::vector<DetectionResult> vec(history_.begin(), history_.end());
    if (Reporter::exportBatchJSON(vec, path.toStdString())) {
        log(QString("Exported %1 records: %2").arg(vec.size()).arg(QFileInfo(path).fileName()));
        statusBar()->showMessage(
            QString("Exported %1 records  ·  %2").arg(vec.size()).arg(path), 5000);
    } else {
        QMessageBox::critical(this, "Export Failed", "Cannot write:\n" + path);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  HISTORY
// ═══════════════════════════════════════════════════════════════════════════

void MainWindow::addHistory(const DetectionResult &r, qint64 ms)
{
    history_.append(r);
    int row = ui->tblHistory->rowCount();
    ui->tblHistory->insertRow(row);

    QColor  vc  = r.is_defect ? QColor(C_RED) : QColor(C_GREEN);

    auto mk = [](const QString &t, QColor c) {
        auto *it = new QTableWidgetItem(t);
        it->setForeground(c);
        it->setFlags(it->flags() & ~Qt::ItemIsEditable);
        return it;
    };

    ui->tblHistory->setItem(row, 0, mk(QString::number(r.frame_id), QColor(C_DIM)));
    ui->tblHistory->setItem(row, 1, mk(QString::fromStdString(r.timestamp), QColor(C_DIM)));
    ui->tblHistory->setItem(row, 2, mk(QFileInfo(QString::fromStdString(r.image_path)).fileName(), QColor(C_TEXT)));
    ui->tblHistory->setItem(row, 3, mk(r.is_defect ? "DEFECT" : "PASS", vc));
    ui->tblHistory->setItem(row, 4, mk(QString::number((double)r.anomaly_score, 'f', 2), vc));
    ui->tblHistory->setItem(row, 5, mk(QString("X%1  Y%2").arg(r.defect_location.x).arg(r.defect_location.y), QColor(C_TEXT)));
    ui->tblHistory->setItem(row, 6, mk(ms > 0 ? QString("%1 ms").arg(ms) : "—", QColor(C_DIM)));
    ui->tblHistory->scrollToBottom();
    ui->lblHistCount->setText(QString("%1 records").arg(history_.size()));
    QMetaObject::invokeMethod(this, [this]() {
        ui->btnClearHistory->setEnabled(true);
        ui->btnExportAll->setEnabled(true);
    }, Qt::QueuedConnection);
}

void MainWindow::onClearHistory()
{
    if (QMessageBox::question(this, "Clear History",
            "Clear all inspection records?",
            QMessageBox::Yes | QMessageBox::No) != QMessageBox::Yes) return;
    history_.clear();
    ui->tblHistory->setRowCount(0);
    ui->lblHistCount->setText("0 records");
    ui->btnClearHistory->setEnabled(false);
    ui->btnExportAll->setEnabled(false);
    ui->lblBatchSummary->setText(" ");
    if (detector_) detector_->resetFrameCounter();
    log("History cleared.");
}

void MainWindow::onExportAll()
{
    if (history_.isEmpty()) {
        QMessageBox::information(this, "Export All", "No records to export.");
        return;
    }
    QString date    = QDateTime::currentDateTime().toString("yyMMdd");
    QString defName = QString("%1_sehfeld_session.json").arg(date);
    QString path = QFileDialog::getSaveFileName(
        this, "Export All Records", defName, "JSON (*.json)");
    if (path.isEmpty()) return;

    std::vector<DetectionResult> vec(history_.begin(), history_.end());
    if (Reporter::exportBatchJSON(vec, path.toStdString())) {
        log(QString("Exported %1 records: %2").arg(vec.size()).arg(QFileInfo(path).fileName()));
        statusBar()->showMessage(
            QString("Exported %1 records  ·  %2").arg(vec.size()).arg(path), 5000);
    } else {
        QMessageBox::critical(this, "Export Failed", "Cannot write:\n" + path);
    }
}