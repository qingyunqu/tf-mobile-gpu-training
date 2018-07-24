//
// Created by WangYingnan on 3/12/17.
//

#ifndef RSKERNELSTEST_RSMATMUL_H
#define RSKERNELSTEST_RSMATMUL_H

#include "tensorflow/contrib/android_renderscript_ops/jni/RScommon.h"
#include "tensorflow/contrib/android_renderscript_ops/utils/android_utils.h"
#include <time.h>

namespace androidrs {

namespace matmul {

static sp<RS> mRS = new RS();
static sp<ScriptIntrinsicBLAS> sc = nullptr; //ScriptIntrinsicBLAS::create(androidrs::matmul::mRS);
//static const char* cachePath = "/data/user/0/org.tensorflow.demo/cache";
static int tot_matmul_cnt = 26;
static int count = 0;
static std::vector<sp<Allocation>> a_alloc_vec;
static std::vector<sp<Allocation>> b_alloc_vec;
static std::vector<sp<Allocation>> c_alloc_vec;

sp<ScriptIntrinsicBLAS>& initSC()
{
    
    if (sc == nullptr) {
    mRS->init(kCachePath);
    sc = ScriptIntrinsicBLAS::create(androidrs::matmul::mRS);   
    }
    return sc;
}

// float
void rsMatmul_sgemm(void* a_ptr, bool a_trans, void* b_ptr, bool b_trans, void* c_ptr,
                    int m, int k,int l, int n, float alpha, float beta)
{
    int idx = count%tot_matmul_cnt;
	int a_m,a_k,b_l,b_n;
	a_m = m;
	a_k = k;
	b_l = l;
	b_n = n;
	if(a_trans){
		a_m = k;
		a_k = m;
	}
	if(b_trans){
		b_l = n;
		b_n = l;
	}
	//printf("[%d,%d]*[%d,%d]\n",a_m,a_k,b_l,b_n);
    if(!androidrs::matmul::mRS->getContext()){
        androidrs::matmul::mRS->init(kCachePath);
    }
//android_log_print("begin create");
//    if(count<tot_matmul_cnt){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);

        sp<const Type> a_t = Type::create(androidrs::matmul::mRS, e, k, m, 0);
        sp<const Type> b_t = Type::create(androidrs::matmul::mRS, e, n, l, 0);
        sp<const Type> c_t = Type::create(androidrs::matmul::mRS, e, b_n, a_m, 0);

        sp<Allocation> a_alloc = Allocation::createTyped(androidrs::matmul::mRS, a_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        sp<Allocation> b_alloc = Allocation::createTyped(androidrs::matmul::mRS, b_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        sp<Allocation> c_alloc = Allocation::createTyped(androidrs::matmul::mRS, c_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);

//        a_alloc_vec.push_back(a_alloc);
//        b_alloc_vec.push_back(b_alloc);
//        c_alloc_vec.push_back(c_alloc);
//     }

//	timespec start,finish;
//	clock_gettime(CLOCK_MONOTONIC, &start);

//    a_alloc_vec[idx]->copy2DRangeFrom(0, 0, k, m, a_ptr);
//    b_alloc_vec[idx]->copy2DRangeFrom(0, 0, n, l, b_ptr);
//android_log_print("begin copy2D");
    a_alloc->copy2DRangeFrom(0, 0, k, m, a_ptr);
    b_alloc->copy2DRangeFrom(0, 0, n, l, b_ptr);

//	clock_gettime(CLOCK_MONOTONIC, &finish);
//	float copy_time = (finish.tv_sec - start.tv_sec) + ((float)(finish.tv_nsec - start.tv_nsec)/1000000000.0f);
//	printf("a b copy time: %f\t",copy_time);

    RsBlasTranspose a_transpose = a_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;
    RsBlasTranspose b_transpose = b_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;

    sp<ScriptIntrinsicBLAS> script = initSC();
	
//	timespec start_,finish_;	
//	clock_gettime(CLOCK_MONOTONIC,&start_);
//     script->SGEMM(a_transpose, b_transpose, alpha, a_alloc_vec[idx], b_alloc_vec[idx], beta, c_alloc_vec[idx]);
//	clock_gettime(CLOCK_MONOTONIC,&finish_);
//	float sgemm_time = (finish_.tv_sec - start_.tv_sec) + ((float)(finish_.tv_nsec - start_.tv_nsec)/1000000000.0f);
//	printf("sgemm time: %f\n",sgemm_time);
	android_log_print("SGEMM");
    script->SGEMM(a_transpose, b_transpose, alpha, a_alloc, b_alloc, beta, c_alloc);

    //c_alloc_vec[idx]->copy2DRangeTo(0, 0, b_n, a_m, c_ptr);
    c_alloc->copy2DRangeTo(0, 0, b_n, a_m, c_ptr);
    count++;
};

}
}

#endif //RSKERNELSTEST_RSMATMUL_H
