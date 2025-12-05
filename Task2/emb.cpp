#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h> 
#include <cstdlib>

using namespace std;
using namespace std::chrono;

const int embedding_table_size = 1000000;
const int embedding_dim = 128;
const int input_size = 720;
const int num_bags = 20;


int random_int(int range) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, range - 1);
    return dis(gen);
}

long long run_with_prefetching( vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
      vector<vector<float>> output;
    int j_prefetch_distance = 32;
    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();
        vector<float> bag_embedding(embedding_dim, 0.0f);
        // if (i + i_prefetch_distance < offsets.size()) {
        //     _mm_prefetch(&offsets[i+i_prefetch_distance], _MM_HINT_T2);
        // }
        for (int j = start_idx; j < end_idx; ++j) {
            // if (j + j_prefetch_distance < end_idx) {
            //     _mm_prefetch(&embedding_table[input[j + j_prefetch_distance] * embedding_dim], _MM_HINT_T0);
            // }
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
            if(d + j_prefetch_distance < embedding_dim){
                _mm_prefetch(&bag_embedding[d + j_prefetch_distance] , _MM_HINT_T0);
                _mm_prefetch(&data_ptr[d + j_prefetch_distance] , _MM_HINT_T0);
            }
                bag_embedding[d] += data_ptr[d];
            }
        }
        output.push_back(bag_embedding);
    }
    
    //-------------------------------------------------------------------------------------------------------------------------------------------
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_simd( vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
     vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; d = d + 4) {
                __m128 bag = _mm_loadu_ps(&bag_embedding[d]);							
                __m128 flt = _mm_loadu_ps(&data_ptr[d]);	
                bag = _mm_add_ps(bag, flt);
                _mm_storeu_ps(&bag_embedding[d],bag);		
                // bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
    }
    
    //-------------------------------------------------------------------------------------------------------------------------------------------
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH SIMD: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_prefetching_simd( vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    
     
     vector<vector<float>> output;
    // _mm_prefetch(&offsets[16],_MM_HINT_T1);
    // _mm_prefetch(&offsets[32],_MM_HINT_T1);
    int j_prefetch_distance = 32;
    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();
        vector<float> bag_embedding(embedding_dim, 0.0f);
        // if (i + i_prefetch_distance < offsets.size()) {
        //     _mm_prefetch(&offsets[i+i_prefetch_distance], _MM_HINT_T2);
        // }
        for (int j = start_idx; j < end_idx; ++j) {
            // if (j + j_prefetch_distance < end_idx) {
            //     _mm_prefetch(&embedding_table[input[j + j_prefetch_distance] * embedding_dim], _MM_HINT_T0);
            // }
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; d = d + 4) {
                _mm_prefetch(&bag_embedding[d + j_prefetch_distance] , _MM_HINT_T2);
                _mm_prefetch(&data_ptr[d + j_prefetch_distance] , _MM_HINT_T2);

                __m128 bag = _mm_loadu_ps(&bag_embedding[d]);							
                __m128 flt = _mm_loadu_ps(&data_ptr[d]);	
                bag = _mm_add_ps(bag, flt);
                _mm_storeu_ps(&bag_embedding[d],bag);	
            }
        }
        output.push_back(bag_embedding);
    }
    //-------------------------------------------------------------------------------------------------------------------------------------------
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching and SIMD: " << duration.count() << " microseconds.";

    return duration.count();
}


long long naive_emb(vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITHOUT software prefetching: " << duration.count() << " microseconds.";
    
    return duration.count();
}

int main() {
    // Prepare embedding table
    vector<float> embedding_table(embedding_table_size * embedding_dim);
    for (auto& val : embedding_table) {
        val = static_cast<float>(random_int(embedding_table_size));
    }

    // Input indices
    vector<int> input(input_size);
    for (auto& idx : input) {
        idx = random_int(embedding_table_size);
    }

    // Offsets
    vector<int> offsets;
    for (int i = 0; i < num_bags; ++i) {
        offsets.push_back((input_size * i) / num_bags);
    }

    // Run naive code 
    long long time_without_prefetch = naive_emb(embedding_table, input, offsets);
    
    // ---------- Flush Cache Before Running Prefetching ----------
    for (size_t i = 0; i < embedding_table.size(); i += 16) {
        _mm_clflush(&embedding_table[i]);
    }
    _mm_mfence();
    
    // Run emb with software prefetching 
    long long time_with_prefetch = run_with_prefetching(embedding_table, input, offsets);
    // Run emb with simd 
    long long time_with_simd = run_with_simd(embedding_table, input, offsets);
    // Run emb with software prefetching and simd
    long long time_with_prefetch_simd = run_with_prefetching_simd(embedding_table, input, offsets);
    

    // Compute speedup
    double speedup1 = static_cast<double>(time_without_prefetch) / time_with_prefetch;
    double speedup2 = static_cast<double>(time_without_prefetch) / time_with_simd;
    double speedup3 = static_cast<double>(time_without_prefetch) / time_with_prefetch_simd;
    cout << fixed << setprecision(3);
    cout << "\n\nSpeedup (with software prefetching) = " << speedup1 << "x\n";
    cout << "Speedup (with simd) = " << speedup2 << "x\n";
    cout << "Speedup (with software prefetching and simd) = " << speedup3 << "x\n";

    return 0;
}

