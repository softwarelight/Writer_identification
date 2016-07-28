#ifndef INTEDIO_CORE_SCOPE_TIMER_HPP
#define INTEDIO_CORE_SCOPE_TIMER_HPP
#include <stdint.h>
#include <string>
//#include <glog/logging.h>
#include <fstream>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

class ScopeTimer {
public:    
    ScopeTimer(const char* proc_desc, uint64_t* ptv_us=NULL)
        : _ptv(ptv_us)
    {
        if (proc_desc) {
            _proc_desc = proc_desc;
        }
        if (_ptv || !_proc_desc.empty()) {
            _start = boost::posix_time::microsec_clock::universal_time();
        }
    }

    ~ScopeTimer()
    {
        if (!_ptv && _proc_desc.empty()) {
            // nothing to note
            return;
        }
        // calc time interval
        boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();
        boost::posix_time::millisec_posix_time_system_config::time_duration_type time_elpse =
                now - _start;
        uint64_t tv_us = time_elpse.ticks();

        // set ptv_us
        if (_ptv) {
            *_ptv = tv_us;
        }
        // log
        double tv_ms = tv_us*1.0/1000;
        if (!_proc_desc.empty()) {
          //  LOG(INFO) << _proc_desc <<"_"<<boost::this_thread::get_id()<< " cost time:" << tv_ms << "(ms)";
            std::fstream fout("log_time.txt", std::ios::out);
            std::cout << _proc_desc <<"_"<<boost::this_thread::get_id()<< " cost time:" << tv_ms << "(ms)"<<std::endl;  
            fout << _proc_desc <<"_"<<boost::this_thread::get_id()<< " cost time:" << tv_ms << "(ms)"<<std::endl;       
            fout.close();
        }
    }

private:
    std::string _proc_desc;  // 输出处理时长日志时添加的描述　
    uint64_t* _ptv;          // 将处理时长透传到外部
    boost::posix_time::ptime _start;
    // disable
    ScopeTimer(const ScopeTimer&);
    ScopeTimer& operator = (const ScopeTimer&);
};

#endif  // INTEDIO_CORE_SCOPE_TIMER_HPP
