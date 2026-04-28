#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QListWidget>
#include <QTableWidget>
#include <QTimer>
#include <QElapsedTimer>
#include <QFutureWatcher>
#include <QScreen>
#include <QGuiApplication>
#include <opencv2/core.hpp>
#include "src/AnomalyDetector.hpp"
#include "src/Reporter.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void resizeEvent(QResizeEvent* event) override;

private slots:
    void onTabLive();
    void onTabBatch();
    void onTabHistory();
    void onLoadImage();
    void onRunDetection();
    void onExportReport();
    void onDetectionFinished();
    void onBatchLoad();
    void onBatchRun();
    void onBatchExport();
    void onExportAll();
    void onClearHistory();
    void updateClock();

private:
    Ui::MainWindow* ui;

    AnomalyDetector* detector_ = nullptr;
    cv::Mat                currentImage_;
    QString                currentImagePath_;
    DetectionResult        lastResult_;
    QStringList            batchFiles_;
    QString                batchFolderName_;
    QList<DetectionResult> history_;

    int sampleCount_ = 0;
    int passCount_ = 0;
    int defectCount_ = 0;

    QFutureWatcher<DetectionResult>* watcher_ = nullptr;
    QElapsedTimer  elapsedTimer_;
    QTimer* clockTimer_ = nullptr;

    void setupStyles();
    void setupDetector();
    void setupCategories();
    void log(const QString& msg);
    void setTabActive(int idx);
    void displayImage(QLabel* lbl, const cv::Mat& img);
    void showResult(const DetectionResult& r, qint64 ms);
    void updateSessionStats();
    void setStripState(bool isDefect);
    void addHistory(const DetectionResult& r, qint64 ms);
    QPixmap matToPixmap(const cv::Mat& img) const;

    // DPI-aware sizes
    float sc_ = 1.0f;
    int H_HEADER_ = 46;
    int W_LEFT_ = 240;
    int H_STRIP_ = 118;
    int H_FEED_HDR_ = 26;
    int PAD_TAB_ = 18;
    int PAD_CONTENT_ = 8;
    int FONT_BASE_ = 12;
    int FONT_SMALL_ = 9;
    int FONT_MONO_ = 11;
};