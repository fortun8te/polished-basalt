"""Host fault injection of the actual adaptive publication code, without CUDA."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/program/generate.cpp'


def between(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


class AdaptiveTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
        if not compiler:
            raise unittest.SkipTest('host C++ compiler unavailable')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        text = SOURCE.read_text()
        helper = '[[noreturn]] void adaptive_transfer_failed' + between(text, '[[noreturn]] void adaptive_transfer_failed', '// " (N MiB')
        # Use the actual CLI pending-publication lambda and deferred-copy callback.
        cli = text.rsplit('        auto apply_pending = [&](bool wait) {', 1)[1].split('        // The VRAM tier follows', 1)[0]
        cli = 'auto apply_pending = [&](bool wait) {' + cli
        callback = between(text, 'ver.set_post_launch([&, pw] {', '                    });')
        harness = r'''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
using cudaError_t = int;
using cudaEvent_t = void*;
constexpr int cudaSuccess=0, cudaErrorNotReady=1, cudaErrorInvalidValue=3;
constexpr int cudaMemcpyHostToDevice=0;
int injected=0, copies=0;
const char* cudaGetErrorString(int) { return "injected"; }
int cudaEventSynchronize(void*) { return injected; }
int cudaEventQuery(void*) { return injected; }
int cudaEventRecord(void*, void*) { std::puts("record"); return injected==6 ? 2 : 0; }
int cudaMemcpyAsync(void*, const void*, size_t, int, void*) {
  ++copies; std::printf("copy%d\n",copies);
  return (injected==4 && copies==1) || (injected==5 && copies==2) ? 2 : 0;
}
namespace basalt {
namespace log { FILE* stream(){return stderr;} void error(const char*, const char*, const char* a, const char*){std::fprintf(stderr,"failure:%s\n",a);} }
namespace core { void release_gpu_waits(FILE*){std::puts("release_waits");} }
namespace kernels::cpu { struct Layout { size_t blob_bytes(int)const{return 16;} }; const Layout& expert_layout(){static Layout l; return l;} }
}
struct Src {
 void commit_exchanges(){std::puts("commit");}
 void release(int,int){std::puts("release_ram");}
 const uint8_t* blob(int,int){static uint8_t b[16]; return injected==3 ? nullptr:b;}
};
struct Cache { void* device_slot(int){static uint8_t d[16];return d;} };
struct Ver { uint32_t* landed_words(){static uint32_t w[4];return w;} };
'''
        harness += helper
        harness += r'''
int main(int, char**argv) {
 injected=std::atoi(argv[2]);
 Src src; Src* srcp=&src;
 std::vector<int> host_res(1,-1);
 std::vector<std::pair<int,int>> pending{{0,7}};
 struct {int n_expert=1; int n_layers=2;} g;
 void* adapt_ev=nullptr; void* adapt_stream=nullptr;
 int dr=0; int* d_res=&dr;
 auto res_put=[](int*){std::puts("publish_device");return injected==7 ? 2:0;};
'''
        harness += cli
        harness += r'''
 if(argv[1][0]=='a') {
   apply_pending(argv[3][0]=='1');
   std::printf("resident=%d pending=%zu\n",host_res[0],pending.size());
 } else {
   struct XSwap {int layer=0,in=0,slot=0;bool p2p=false;};
   std::vector<XSwap> xq(1); Cache xcache; Ver ver;
   uint32_t words[3]{}; const uint32_t* pw=words;
   auto deferred=[&,pw] {
'''
        harness += callback
        harness += r'''
   };
   deferred(); std::puts("callback_complete");
 }
 std::puts("DONE");
 return 0;
}
'''
        path = Path(cls.tmp.name)
        (path/'test.cpp').write_text(harness)
        cls.exe = path/'test'
        subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror', str(path/'test.cpp'), '-o', str(cls.exe)], check=True, text=True)

    def run_case(self, mode, error, wait=True):
        r = subprocess.run([str(self.exe),mode,str(error),str(int(wait))],capture_output=True,text=True)
        if r.returncode != 0:
            self.assertNotIn('DONE', r.stdout)
        return r

    def test_success_publishes(self):
        r=self.run_case('apply',0)
        self.assertEqual(r.returncode,0)
        self.assertIn('resident=7 pending=0',r.stdout)

    def test_pending_query_keeps_old_residency(self):
        r=self.run_case('apply',1,False)
        self.assertEqual(r.returncode,0)
        self.assertIn('resident=-1 pending=1',r.stdout)
        self.assertNotIn('commit',r.stdout)

    def test_wait_and_query_errors_never_publish(self):
        for wait in (True,False):
            with self.subTest(wait=wait):
                r=self.run_case('apply',2,wait)
                self.assertEqual(r.returncode,1)
                self.assertIn('release_waits',r.stdout)
                self.assertNotIn('commit',r.stdout)
                self.assertNotIn('release_ram',r.stdout)
                self.assertNotIn('publish_device',r.stdout)

    def test_residency_upload_failure_retires_process(self):
        r=self.run_case('apply',7)
        self.assertEqual(r.returncode,1)
        self.assertIn('release_waits',r.stdout)
        self.assertNotIn('resident=',r.stdout)

    def test_callback_success_preserves_order(self):
        r=self.run_case('callback',0)
        self.assertEqual(r.returncode,0)
        self.assertEqual(r.stdout,'copy1\ncopy2\nrecord\ncallback_complete\nDONE\n')

    def test_missing_source_and_copy_failure_do_not_publish_word(self):
        for error in (3,4):
            with self.subTest(error=error):
                r=self.run_case('callback',error)
                self.assertEqual(r.returncode,1)
                self.assertIn('release_waits',r.stdout)
                self.assertNotIn('copy2',r.stdout)
                self.assertNotIn('record',r.stdout)

    def test_landed_word_and_event_record_failures_retire(self):
        for error in (5,6):
            with self.subTest(error=error):
                r=self.run_case('callback',error)
                self.assertEqual(r.returncode,1)
                self.assertIn('release_waits',r.stdout)
                self.assertNotIn('callback_complete',r.stdout)


if __name__ == '__main__': unittest.main()
