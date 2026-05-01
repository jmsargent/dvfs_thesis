#ifndef MUSTARD_STRIDEDPANEL_H
#define MUSTARD_STRIDEDPANEL_H
#include <host/nvshmem_api.h>

#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

class StridedDevicePanel;

class StridedHostPanel {
public:
    StridedHostPanel(int myPE, int nPEs, int N, int B, int T)
        : myPE_(myPE), nPEs_(nPEs), N_(N), B_(B), T_(T)
    {
        for (int k = myPE; k < T; k += nPEs) myNumCols_++;
        myMatrixSize_ = (size_t)myNumCols_ * B * N;
        matrix_       = new double[myMatrixSize_];
    }

    ~StridedHostPanel() { delete[] matrix_; }

    /*
        fills panel with deterministically random numbers
    */
    void fill()
    {
        for (int blockCol = myPE_; blockCol < T_; blockCol += nPEs_)
        {
            int localIdx = blockCol / nPEs_;
            for (int bc = 0; bc < B_; bc++)
            {
                int globalCol = blockCol * B_ + bc;
                for (int row = 0; row < N_; row++)
                {
                    int    d   = std::abs(row - globalCol);
                    double val = (d == 0) ? (double)N_ + 1.0 : 1.0 / (double)(d + 1);
                    matrix_[(size_t)localIdx * B_ * N_ + bc * N_ + row] = val;
                }
            }
        }
    }

    void copyToDevicePanel(StridedDevicePanel& panel);

    bool verify(StridedDevicePanel& panel);

private:
    int     myPE_ = 0, nPEs_, N_, B_, T_;
    int     myNumCols_ = 0;
    size_t  myMatrixSize_;
    double* matrix_;

};

class StridedDevicePanel {
public:
    StridedDevicePanel(double* d_matrices, int PE, int nPEs, int N, int B, int T)
        : nPEs(nPEs), B(B), N(N), device_matrix((double*)nvshmem_ptr(d_matrices, PE)) {}

    StridedDevicePanel(double* device_matrix, int nPEs, int N, int B)
        : nPEs(nPEs), B(B), N(N), device_matrix(device_matrix) {}

    ~StridedDevicePanel() { if (device_matrix) nvshmem_free(device_matrix); }

    StridedDevicePanel(const StridedDevicePanel&)            = delete;
    StridedDevicePanel& operator=(const StridedDevicePanel&) = delete;
    StridedDevicePanel(StridedDevicePanel&& o) noexcept
        : nPEs(o.nPEs), B(o.B), N(o.N), device_matrix(o.device_matrix)
    {
        o.device_matrix = nullptr;
    }

    void    release()          { device_matrix = nullptr; }
    double* tile(int i, int j) { return device_matrix + i * B + (j / nPEs) * B * N; }

private:
    int     nPEs, B, N;
    double* device_matrix;

    friend class StridedHostPanel;
};

inline void StridedHostPanel::copyToDevicePanel(StridedDevicePanel& panel)
{
    cudaMemcpy(panel.device_matrix, matrix_, myMatrixSize_ * sizeof(double), cudaMemcpyHostToDevice);
}

inline bool StridedHostPanel::verify(StridedDevicePanel& panel)
{
    auto tmp = std::make_unique<double[]>(myMatrixSize_);
    cudaMemcpy(tmp.get(), panel.device_matrix, myMatrixSize_ * sizeof(double), cudaMemcpyDeviceToHost);

    for (int j = myPE_; j < T_; j += nPEs_)
    {
        int localJ = j / nPEs_;
        for (int c = 0; c < B_; c++)
            if (tmp[(size_t)localJ * B_ * N_ + j * B_ + c * N_ + c] <= 0.0)
                return false;
    }
    return true;
}


class StridedDevicePanels
{
public:
    StridedDevicePanels(int myPE, int nPEs, int N, int B, int T)
        : myPE_(myPE)
    {
        size_t  maxCols    = ((size_t)T + nPEs - 1) / nPEs;
        double* d_matrices = (double*)nvshmem_malloc(maxCols * B * N * sizeof(double));
        for (int pe = 0; pe < nPEs; pe++)
            panels_.emplace_back(d_matrices, pe, nPEs, N, B, T);
    }

    ~StridedDevicePanels()
    {
        for (int pe = 0; pe < (int)panels_.size(); pe++)
            if (pe != myPE_) panels_[pe].release();
    }

    StridedDevicePanels(StridedDevicePanels&&)                 = default;
    StridedDevicePanels(const StridedDevicePanels&)            = delete;
    StridedDevicePanels& operator=(const StridedDevicePanels&) = delete;

    StridedDevicePanel& myPanel()          { return panels_[myPE_]; }
    StridedDevicePanel& otherPanel(int pe) { return panels_[pe]; }

private:
    int                          myPE_;
    std::vector<StridedDevicePanel> panels_;
};

#endif // MUSTARD_STRIDEDPANEL_H
