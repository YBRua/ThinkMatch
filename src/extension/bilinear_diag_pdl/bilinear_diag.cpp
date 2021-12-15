#include "paddle/extension.h"
#include <iostream>

/* CUDA Declaration */

std::vector<paddle::Tensor> bilinear_diag_csc_cuda(
    const paddle::Tensor &t1_indices,
    const paddle::Tensor &t1_indptr,
    const paddle::Tensor &t1_data,
    const paddle::Tensor &t2,
    const paddle::Tensor &t3_indices,
    const paddle::Tensor &t3_indptr,
    const paddle::Tensor &t3_data,
    int64_t batch_size,
    int64_t xlen);


#define CHECK_CUDA(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")
#define CHECK_CPU(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")
#define CHECK_INPUT(x) CHECK_CUDA(x)


/* Dense Implementation */

// std::vector<paddle::Tensor> bilinear_diag_dense(
//     const paddle::Tensor &t1,
//     const paddle::Tensor &t2,
//     const paddle::Tensor &t3
// ){
//     auto sizes = t1.shape();
//     auto batch_size = sizes[0];
//     auto xlen = sizes[1];
//     auto outp = paddle::Tensor(t2.place(), {batch_size, xlen});
//     for(int64_t i = 0; i < xlen; i++)
//     {
//         auto _t1 = paddle::slice(t1, 1, i, i+1);
//         paddle::
//         auto tmp = paddle::bmm(_t1, t2);
//         auto _t3 = paddle::slice(t3, 2, i, i+1);
//         auto _outp = paddle::bmm(tmp, _t3).view(-1);
//         for(int64_t j = 0; j < batch_size; j++)
//             outp[j][i] = _outp[j];
//     }
//     return outp;
// }


/* COO Sparse Implementation */

bool sort_smaller_than(int64_t main1, int64_t main2, int64_t minor1, int64_t minor2)
{
    if (main1 < main2)
        return true;
    else if (main1 > main2)
        return false;
    else if (minor1 < minor2)
        return true;
    else
        return false;
}


void sort_sparse_helper(const paddle::Tensor &main,
                        const paddle::Tensor &minor,
                        std::vector<paddle::Tensor> others,
                        int64_t begin,
                        int64_t end)
{
    if (begin >= end)
        return;

    auto head = begin;
    auto tail = end;
    auto reverse = true;
    auto* main_access = main.mutable_data<int64_t>();
    auto* minor_access = minor.mutable_data<int64_t>();
    while (head != tail)
    {
        if (sort_smaller_than(main_access[tail], main_access[head], minor_access[tail], minor_access[head]))
        {
            //swap
            std::swap(main_access[head], main_access[tail]);
            std::swap(minor_access[head], minor_access[tail]);
            for (auto iter = others.cbegin(); iter != others.cend(); iter++)
            {
                if (iter->dtype() == paddle::ScalarType::Float)
                {
                    auto others_access = iter->mutable_data<float>();
                    std::swap(others_access[head], others_access[tail]);
                }
                else if (iter->dtype() == paddle::ScalarType::Double)
                {
                    auto others_access = iter->mutable_data<double>();
                    std::swap(others_access[head], others_access[tail]);
                }
                else
                {
                    auto others_access = iter->mutable_data<int64_t>();
                    std::swap(others_access[head], others_access[tail]);
                }
            }
            reverse = !reverse;
        }
        else
        {
            if (reverse)
                tail--;
            else
                head++;
        }
    }

    auto split = head;
    sort_sparse_helper(main, minor, others, begin, split - 1);
    sort_sparse_helper(main, minor, others, split + 1, end);
}


std::vector<paddle::Tensor> sort_sparse(
    const paddle::Tensor &ts,
    int64_t main_dim,
    int64_t minor_dim
)
{
    // assert(ts.is_sparse());
    auto max_dim = ts.dim();

    if (main_dim < 0)
        main_dim += max_dim;
    if (minor_dim < 0)
        minor_dim += max_dim;
    assert(0 <= main_dim && main_dim < max_dim);
    assert(0 <= minor_dim && minor_dim < max_dim);
    assert(main_dim != minor_dim);

    auto ind = ts._indices();
    auto data = ts._values();

    auto ind_sizes = ind.sizes();
    auto dim_len = ind_sizes[1];

    std::vector<paddle::Tensor> others;
    for (int64_t i = 0; i < max_dim; i++)
        if ((i != main_dim) && (i != minor_dim))
            others.push_back(ind[i]);

    others.push_back(data);

    sort_sparse_helper(ind[main_dim], ind[minor_dim], others, 0, dim_len - 1);

    return ts;
}


void split_sorted_coo(paddle::Tensor t, int64_t xlen_indices[], int64_t xlen_dim)
{
    auto indices = t._indices();
    auto t_nnz = t._nnz();
    auto indices_access = indices.accessor<int64_t, 2>();

    int64_t cur_batch = 0;
    int64_t cur_xlen = 0;
    int64_t xlen_offset = 0;
    for (int64_t i = 0; i < t_nnz;)
    {
        if (indices_access[xlen_dim][i] != cur_xlen)
        {
            xlen_indices[xlen_offset + ++cur_xlen] = i;
        }
        if (indices_access[0][i] != cur_batch)
        {
            xlen_offset = cur_xlen;
            cur_xlen = 0;
        }
        if ((indices_access[xlen_dim][i] == cur_xlen) && (indices_access[0][i] == cur_batch))
            i++;
    }
    xlen_indices[xlen_offset + ++cur_xlen] = t_nnz;
}


std::vector<paddle::Tensor> bilinear_diag_coo(
    paddle::Tensor t1,
    paddle::Tensor t2,
    paddle::Tensor t3
){
    auto t1_sizes = t1.sizes();
    auto batch_size = t1_sizes[0];
    auto xlen = t1_sizes[1];
    auto feat_size = t1_sizes[2];
    auto outp = paddle::zeros({batch_size, xlen}, t2.type());

    auto t1_indices = t1._indices();
    auto t1_values = t1._values();
    auto t3_indices = t3._indices();
    auto t3_values = t3._values();

    int64_t t1_xlen_indices[xlen * batch_size + 1] = {0};
    int64_t t3_xlen_indices[xlen * batch_size + 1] = {0};

    auto t1_idx_access = t1_indices.accessor<int64_t, 2>();
    auto t3_idx_access = t3_indices.accessor<int64_t, 2>();

    split_sorted_coo(t1, t1_xlen_indices, 1);
    split_sorted_coo(t3, t3_xlen_indices, 2);

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < xlen; i++)
        {
            auto t1_start = t1_xlen_indices[b * xlen + i];
            auto t1_stop = t1_xlen_indices[b * xlen + i + 1];
            auto t3_start = t3_xlen_indices[b * xlen + i];
            auto t3_stop = t3_xlen_indices[b * xlen + i + 1];

            for (auto t1_idx = t1_start; t1_idx < t1_stop; t1_idx++)
            {
                for (auto t3_idx = t3_start; t3_idx < t3_stop; t3_idx++)
                {
                    outp[b][i] += t2[b][t1_idx_access[2][t1_idx]][t3_idx_access[1][t3_idx]]
                                  * t1_values[t1_idx] * t3_values[t3_idx];
                }
            }

        }
    }
    return outp;
}


/* CSC Sparse Implementation */

std::vector<paddle::Tensor> bilinear_diag_csc_cpu(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2,
    paddle::Tensor t3_indices,
    paddle::Tensor t3_indptr,
    paddle::Tensor t3_data,
    int64_t batch_size,
    int64_t xlen
){
    CHECK_CPU(t1_indices);
    CHECK_CPU(t1_indptr);
    CHECK_CPU(t1_data);
    CHECK_CPU(t2);
    CHECK_CPU(t3_indices);
    CHECK_CPU(t3_indptr);
    CHECK_CPU(t3_data);

    auto outp = paddle::zeros({batch_size, xlen}, t2.type());
    auto t1_indptr_acc = t1_indptr.accessor<int64_t, 1>();
    auto t3_indptr_acc = t3_indptr.accessor<int64_t, 1>();

    for (int64_t b = 0; b < batch_size; b++)
    {
        for (int64_t i = 0; i < xlen; i++)
        {
            int64_t t1_start = t1_indptr_acc[b * xlen + i];
            int64_t t1_stop = t1_indptr_acc[b * xlen + i + 1];
            int64_t t3_start = t3_indptr_acc[b * xlen + i];
            int64_t t3_stop = t3_indptr_acc[b * xlen + i + 1];

            for (auto t1_idx = t1_start; t1_idx < t1_stop; t1_idx++)
            {
                for (auto t3_idx = t3_start; t3_idx < t3_stop; t3_idx++)
                {
                    outp[b][i] += t2[b][t1_indices[t1_idx]][t3_indices[t3_idx]]
                                  * t1_data[t1_idx] * t3_data[t3_idx];
                }
            }

        }
    }
    return outp;
}


std::vector<paddle::Tensor> bilinear_diag_csc_cuda_wrapper(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2,
    paddle::Tensor t3_indices,
    paddle::Tensor t3_indptr,
    paddle::Tensor t3_data,
    int64_t batch_size,
    int64_t xlen
){
    CHECK_INPUT(t1_indices);
    CHECK_INPUT(t1_indptr);
    CHECK_INPUT(t1_data);
    CHECK_INPUT(t2);
    CHECK_INPUT(t3_indices);
    CHECK_INPUT(t3_indptr);
    CHECK_INPUT(t3_data);
    return bilinear_diag_csc_cuda(t1_indices, t1_indptr, t1_data,
                                  t2,
                                  t3_indices, t3_indptr, t3_data,
                                  batch_size, xlen);
}


std::vector<paddle::Tensor> bilinear_diag_csc(
    paddle::Tensor t1_indices,
    paddle::Tensor t1_indptr,
    paddle::Tensor t1_data,
    paddle::Tensor t2,
    paddle::Tensor t3_indices,
    paddle::Tensor t3_indptr,
    paddle::Tensor t3_data,
    int64_t batch_size,
    int64_t xlen
)
{
    if (t1_indices.type().is_cuda())
        return bilinear_diag_csc_cuda_wrapper(t1_indices, t1_indptr, t1_data, t2, t3_indices, t3_indptr, t3_data, batch_size, xlen);

    else
        return bilinear_diag_csc_cpu(t1_indices, t1_indptr, t1_data, t2, t3_indices, t3_indptr, t3_data, batch_size, xlen);
}

/* PyBind Interface */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bilinear_diag", &bilinear_diag_csc, "bilinear diagonal");
}
