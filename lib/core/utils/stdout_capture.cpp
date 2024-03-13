#include "core/utils/stdout_capture.h"
#include <chrono>
#include <thread>

namespace oklt {
StdCapture::StdCapture()
    : m_capturing(false) {
    // make stdout & stderr streams unbuffered
    // so that we don't need to flush the streams
    // before capture and after capture
    // (fflush can cause a deadlock if the stream is currently being
    std::lock_guard<std::mutex> lock(m_mutex);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
}

StdCapture::~StdCapture() {
    EndCapture();
}

void StdCapture::BeginCapture() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_capturing)
        return;

    secure_pipe(m_pipe);
    m_oldStdOut = secure_dup(STD_OUT_FD);
    m_oldStdErr = secure_dup(STD_ERR_FD);
    secure_dup2(m_pipe[WRITE], STD_OUT_FD);
    secure_dup2(m_pipe[WRITE], STD_ERR_FD);
    m_capturing = true;
    secure_close(m_pipe[WRITE]);
}

bool StdCapture::IsCapturing() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_capturing;
}

bool StdCapture::EndCapture() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_capturing)
        return true;

    m_captured.clear();
    secure_dup2(m_oldStdOut, STD_OUT_FD);
    secure_dup2(m_oldStdErr, STD_ERR_FD);

    const int bufSize = 1025;
    char buf[bufSize];
    int bytesRead = 0;
    bool fd_blocked(false);
    do {
        bytesRead = 0;
        fd_blocked = false;
        bytesRead = read(m_pipe[READ], buf, bufSize - 1);
        if (bytesRead > 0) {
            buf[bytesRead] = 0;
            m_captured += buf;
        } else if (bytesRead < 0) {
            fd_blocked = (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR);
            if (fd_blocked)
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } while (fd_blocked || bytesRead == (bufSize - 1));

    secure_close(m_oldStdOut);
    secure_close(m_oldStdErr);
    secure_close(m_pipe[READ]);
    m_capturing = false;
    return true;
}

std::string StdCapture::GetCapture() {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_captured;
}

int StdCapture::secure_dup(int src) {
    int ret = -1;
    bool fd_blocked = false;
    do {
        ret = dup(src);
        fd_blocked = (errno == EINTR || errno == EBUSY);
        if (fd_blocked)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (ret < 0);
    return ret;
}

void StdCapture::secure_pipe(int* pipes) {
    int ret = -1;
    bool fd_blocked = false;
    do {
        ret = pipe(pipes) == -1;
        fd_blocked = (errno == EINTR || errno == EBUSY);
        if (fd_blocked)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (ret < 0);
}

void StdCapture::secure_dup2(int src, int dest) {
    int ret = -1;
    bool fd_blocked = false;
    do {
        ret = dup2(src, dest);
        fd_blocked = (errno == EINTR || errno == EBUSY);
        if (fd_blocked)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (ret < 0);
}

void StdCapture::secure_close(int& fd) {
    int ret = -1;
    bool fd_blocked = false;
    do {
        ret = close(fd);
        fd_blocked = (errno == EINTR);
        if (fd_blocked)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (ret < 0);

    fd = -1;
}
}  // namespace oklt
