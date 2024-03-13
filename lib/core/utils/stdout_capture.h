#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <mutex>

namespace oklt {
#ifndef STD_OUT_FD
#define STD_OUT_FD (fileno(stdout))
#endif

#ifndef STD_ERR_FD
#define STD_ERR_FD (fileno(stderr))
#endif

class StdCapture {
   public:
    StdCapture();
    ~StdCapture();

    void BeginCapture();
    bool IsCapturing();
    bool EndCapture();
    std::string GetCapture();

   private:
    enum PIPES { READ, WRITE };

    int secure_dup(int src);
    void secure_pipe(int* pipes);
    void secure_dup2(int src, int dest);
    void secure_close(int& fd);

    int m_pipe[2];
    int m_oldStdOut;
    int m_oldStdErr;
    bool m_capturing;
    std::mutex m_mutex;
    std::string m_captured;
};
}  // namespace oklt
