#ifndef MUSTARD_STRIDEDPANEL_H
#define MUSTARD_STRIDEDPANEL_H
#include <host/nvshmem_api.h>
#include <vector>

class StridedPanel {
public:
    StridedPanel(double* d_matrices, int PE, int nPEs, int N, int B, int T)
        : nPEs(nPEs), B(B), N(N), device_matrix((double*)nvshmem_ptr(d_matrices, PE)) {}

    StridedPanel(double* device_matrix, int nPEs, int N, int B)
        : nPEs(nPEs), B(B), N(N), device_matrix(device_matrix) {}

    ~StridedPanel() { if (device_matrix) nvshmem_free(device_matrix); }

    StridedPanel(const StridedPanel&)            = delete;
    StridedPanel& operator=(const StridedPanel&) = delete;
    StridedPanel(StridedPanel&& o) noexcept
        : nPEs(o.nPEs), B(o.B), N(o.N), device_matrix(o.device_matrix)
    {
        o.device_matrix = nullptr;
    }

    void    release()              { device_matrix = nullptr; }
    double* tile(int i, int j)     { return device_matrix + i * B + (j / nPEs) * B * N; }

private:
    int     nPEs, B, N;
    double* device_matrix;
};

class StridedPanels
{
public:
    StridedPanels(int myPE, int nPEs, int N, int B, int T)
        : myPE_(myPE)
    {
        size_t  maxCols    = ((size_t)T + nPEs - 1) / nPEs;
        double* d_matrices = (double*)nvshmem_malloc(maxCols * B * N * sizeof(double));
        for (int pe = 0; pe < nPEs; pe++)
            panels_.emplace_back(d_matrices, pe, nPEs, N, B, T);
    }

    ~StridedPanels()
    {
        // Only free the panel that is local to the PE, the other PEs
        // take care of their allocation
        for (int pe = 0; pe < (int)panels_.size(); pe++)
            if (pe != myPE_) panels_[pe].release();
    }

    StridedPanel& myPanel()          { return panels_[myPE_]; }
    StridedPanel& otherPanel(int pe) { return panels_[pe]; }

private:
    int                       myPE_;
    std::vector<StridedPanel> panels_;
};

#endif // MUSTARD_STRIDEDPANEL_H
