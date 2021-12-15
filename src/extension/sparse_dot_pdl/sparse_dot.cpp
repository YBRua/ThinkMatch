#include "paddle/extension.h"
#include <utility>

/* CUDA Declaration */

std::vector<paddle::Tensor> csr_dot_csc_cuda(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2_indices,
    const paddle::Tensor &t2_indptr,
    const paddle::Tensor &t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
);


std::vector<paddle::Tensor> csr_dot_diag_cuda(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
);


#define CHECK_CUDA(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")
#define CHECK_CPU(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")
#define CHECK_INPUT(x) CHECK_CUDA(x)


/* CSR dot CSC Implementation */

std::vector<paddle::Tensor> csr_dot_csc_cpu(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2_indices,
    const paddle::Tensor &t2_indptr,
    const paddle::Tensor &t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
){
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2_indices);
    CHECK_CPU(t2_indptr);
    CHECK_CPU(t2_data);

    std::list<int64_t> out_indices_list[batch_size * out_h];
    std::list<float> out_data_list[batch_size * out_h];
    // auto out_indptr = paddle::zeros({batch_size * out_h + 1}, t1_indptr.type());
    auto out_indptr = paddle::Tensor(t1_indptr.place(), {batch_size * out_h + 1});
    auto* out_indptr_acc = out_indptr.mutable_data<int64_t>(t1_indptr.place());
    auto* t1_indptr_acc = t1_indptr.data<int64_t>();
    auto* t2_indptr_acc = t2_indptr.data<int64_t>();
    auto* t1_indices_acc = t1_indices.data<int64_t>();
    auto* t2_indices_acc = t2_indices.data<int64_t>();
    auto* t1_data_acc = t1_data.data<float>();
    auto* t2_data_acc = t2_data.data<float>();

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            int64_t t1_start = t1_indptr_acc[b * out_h + i];
            int64_t t1_stop = t1_indptr_acc[b * out_h + i + 1];
            int64_t row_nnz = 0;

            for (int64_t j = 0; j < out_w; j++)
            {
                int64_t t2_start = t2_indptr_acc[b * out_w + j];
                int64_t t2_stop = t2_indptr_acc[b * out_w + j + 1];

                float outp = 0;
                int64_t t1_ptr_idx = t1_start;
                int64_t t2_ptr_idx = t2_start;

                while (t1_ptr_idx < t1_stop && t2_ptr_idx < t2_stop)
                {
                    int64_t t1_cur_indice = t1_indices_acc[t1_ptr_idx];
                    int64_t t2_cur_indice = t2_indices_acc[t2_ptr_idx];
                    if (t1_cur_indice == t2_cur_indice)
                    {
                        auto tmp = t1_data_acc[t1_ptr_idx] * t2_data_acc[t2_ptr_idx];
                        // auto tmp_acc = tmp.accessor<float, 1>();
                        // outp += tmp_acc[0];
                        outp += tmp;
                        t1_ptr_idx++;
                        t2_ptr_idx++;
                    }
                    else if (t1_cur_indice < t2_cur_indice)
                        t1_ptr_idx++;
                    else
                        t2_ptr_idx++;
                }
                if (outp != 0)
                {
                    out_data_list[b * out_h + i].push_back(outp);
                    out_indices_list[b * out_h + i].push_back(j);
                    row_nnz++;
                }
            }
            out_indptr_acc[b * out_h + i + 1] = out_indptr_acc[b * out_h + i] + row_nnz;
        }
    }

    int64_t nnz = out_indptr_acc[out_indptr.size()-1];
    // auto out_indices = paddle::zeros({nnz}, t1_indices.type());
    auto out_indices = paddle::Tensor(t1_indices.place(), {nnz});
    auto* out_indices_acc = out_indices.mutable_data<int64_t>(t1_indices.place());
    auto out_data = paddle::Tensor(t1_data.place(), {nnz});
    auto* out_data_acc = out_data.mutable_data<float>(t1_data.place());
    int64_t idx = 0;
    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            auto * tmp_indices_list = &out_indices_list[b * out_h + i];
            auto * tmp_data_list = &out_data_list[b * out_h + i];
            while (!tmp_indices_list->empty() && !tmp_data_list->empty())
            {
                out_indices_acc[idx] = tmp_indices_list->front();
                tmp_indices_list->pop_front();
                out_data_acc[idx] = tmp_data_list->front();
                tmp_data_list->pop_front();
                idx++;
            }
        }
    }

    return {out_indices, out_indptr, out_data};
}


std::vector<paddle::Tensor> csr_dot_csc_dense_cuda_wrapper(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2_indices,
    const paddle::Tensor &t2_indptr,
    const paddle::Tensor &t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
){
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2_indices);
    CHECK_INPUT(t2_indptr);
    CHECK_INPUT(t2_data);
    return csr_dot_csc_cuda(t1_indices, t1_indptr, t1_data,
                            t2_indices, t2_indptr, t2_data,
                            batch_size, out_h, out_w);
}


std::vector<paddle::Tensor> csr_dot_csc(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2_indices,
    const paddle::Tensor &t2_data,
    const paddle::Tensor &t2_indptr,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    if (t1_indices.place() == paddle::PlaceType::kGPU)
        throw std::runtime_error("Unexpected cuda tensor in sparse dot sparse -> sparse computation.");
    else
        return csr_dot_csc_cpu(t1_indices, t1_indptr, t1_data, t2_indices, t2_indptr, t2_data, batch_size, out_h, out_w);
}

paddle::Tensor csr_dot_csc_dense_cuda(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2_indices,
    const paddle::Tensor &t2_indptr,
    const paddle::Tensor &t2_data,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    return csr_dot_csc_dense_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2_indices, t2_indptr, t2_data,
                                          batch_size, out_h, out_w);
}


/* CSR dot diag implementation */

std::vector<paddle::Tensor> csr_dot_diag_cpu(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2);
    auto t2_nrows = t2.shape()[0];
    auto outp_indices = t1_indices.copy_to(t1_indices.place());
    auto outp_indptr = t1_indptr.copy_to(t1_indptr.place());
    auto outp_data = paddle::Tensor(t1_data.place(), t1_data.shape());

    auto* t1_indptr_acc = t1_indptr.data<int64_t>();
    auto* t1_indices_acc = t1_indices.data<int64_t>();
    auto* outp_data_ptr = outp_data.mutable_data<float>(t1_indptr.place());


    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < out_h; i++)
        {
            int64_t start = t1_indptr_acc[b * out_h + i];
            int64_t stop = t1_indptr_acc[b * out_h + i + 1];
            for (int64_t data_idx = start; data_idx < stop; data_idx++)
            {
                int64_t row_idx = t1_indices_acc[data_idx];
                outp_data_ptr[data_idx] = t1_data[data_idx] * t2[b * t2_nrows + row_idx];
            }
        }
    }
    return {outp_indices, outp_indptr, outp_data};
}


std::vector<paddle::Tensor> csr_dot_diag_cuda_wrapper(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2);
    return csr_dot_diag_cuda(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);
}


std::vector<paddle::Tensor> csr_dot_diag(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2,
    int64_t batch_size,
    int64_t out_h,
    int64_t out_w
)
{
    if (t1_indices.place() == paddle::PlaceType::kGPU)
        return csr_dot_diag_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);
    else
        // throw std::runtime_error("CPU CSR-dot-Diag is not implemented for now!");
        return csr_dot_diag_cpu(t1_indices, t1_indptr, t1_data, t2, batch_size, out_h, out_w);

}

/* PyBind Interface */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csr_dot_csc", &csr_dot_csc, "csr sparse matrix dot csc sparse matrix");
  m.def("csr_dot_csc_dense_cuda", &csr_dot_csc_dense_cuda,
        "cuda implementation of csr sparse matrix dot csc sparse matrix, result is dense");
  m.def("csr_dot_diag", &csr_dot_diag, "csr sparse matrix dot a diagonal of dense vector");
}
