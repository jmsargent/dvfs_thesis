#pragma once

#include <cstdarg>
#include <cstdio>
#include <string>

// PEWriter routes printf-style output to a per-PE file (<prefix>_pe<N>.csv),
// or to stdout when no prefix is given. Eliminates collisions in multi-GPU
// jobs where all ranks share the same Slurm output file.
//
// Usage:
//   PEWriter out(cfg.outputPrefix, myPE);
//   out.print("pe,run,task_id\n");
//   out.print("%d,%d,%d\n", myPE, run, task_id);
class PEWriter
{
   public:
    PEWriter(const std::string& prefix, int pe) : f_(nullptr), owns_file_(false)
    {
        if (prefix.empty())
        {
            f_          = stdout;
            owns_file_  = false;
        }
        else
        {
            char fname[512];
            snprintf(fname, sizeof(fname), "%s_pe%d.csv", prefix.c_str(), pe);
            f_         = fopen(fname, "w");
            owns_file_ = true;
            if (!f_)
            {
                fprintf(stderr, "PEWriter: could not open '%s'\n", fname);
                f_         = stdout;
                owns_file_ = false;
            }
        }
    }

    ~PEWriter()
    {
        if (owns_file_ && f_)
            fclose(f_);
    }

    // Not copyable; the file handle has a single owner.
    PEWriter(const PEWriter&)            = delete;
    PEWriter& operator=(const PEWriter&) = delete;

    __attribute__((format(printf, 2, 3)))
    void print(const char* fmt, ...)
    {
        va_list args;
        va_start(args, fmt);
        vfprintf(f_, fmt, args);
        va_end(args);
    }

    void flush() { fflush(f_); }

   private:
    FILE* f_;
    bool  owns_file_;
};
