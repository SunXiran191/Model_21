#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <unistd.h>
#include <cassert>
#include <csignal>
#include <vector>
#include "../res/include/video_get.hpp"
#include "../thread/thread.hpp"

using namespace std;
using namespace cv;

atomic<bool> g_exit(false);
atomic<bool> stop_flag(true);
mutex mtx_produce;
Produce g_produce;
atomic<uint64_t> g_produce_seq(0);
atomic<uint64_t> g_seg_seq(0);
atomic<uint64_t> g_render_seq(0);
atomic<double> g_run_thread_fps(0.0);
atomic<float> speed_now(0.0f);
atomic<float> g_error(0.0f);

// Uart uart("/dev/ttyUSB0");

static bool load_config_from_file(const std::string &path, Config &config)
{
	std::ifstream ifs(path.c_str());
	if (!ifs.is_open())
	{
		std::cerr << "warning: config file not found, using defaults: " << path << std::endl;
		return false;
	}

	try
	{
		nlohmann::json j;
		ifs >> j;
		config = j.get<Config>();
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "warning: failed to parse config, using defaults: " << e.what() << std::endl;
		return false;
	}
}

static bool load_entry_config(const std::string &entry_path, std::string &out_config_path)
{
	std::ifstream ifs(entry_path.c_str());
	if (!ifs.is_open())
	{
		std::cerr << "warning: entry config not found: " << entry_path << std::endl;
		return false;
	}
	try
	{
		nlohmann::json j;
		ifs >> j;
		if (j.contains("config_path"))
			out_config_path = j["config_path"].get<std::string>();
		return true;
	}
	catch (const std::exception &e)
	{
		std::cerr << "warning: failed to parse entry config: " << e.what() << std::endl;
		return false;
	}
}

void on_signal(int)
{
	g_exit = true;
}

int main()
{
	Config config;
	// 入口配置文件路径（运行目录为 build/）
	const std::string entry_config_path = "../res/configs/config.json";
	std::string active_config;
	load_entry_config(entry_config_path, active_config);

	// 如果入口文件中指定了 config_path，则加载该配置；否则使用默认配置
	std::string config_path;
	if (!active_config.empty())
	{
		// 相对路径基于 res/configs/ 目录
		if (active_config[0] == '/')
			config_path = active_config;
		else
			config_path = "../res/configs/" + active_config;
	}
	else
	{
		config_path = "../res/configs/config_1.2m.json";
	}

	std::cout << "[main] loading config from: " << config_path << std::endl;
	if (!load_config_from_file(config_path, config))
	{
		std::cerr << "fatal: cannot load config from " << config_path << std::endl;
		return -1;
	}

	std::thread produce_t(&produce_thread, std::ref(g_produce), std::cref(config));
	std::thread infer_det_t(&infer_det_thread, std::ref(g_produce), std::cref(config));
	std::thread infer_seg_t(&infer_seg_thread, std::ref(g_produce), std::cref(config));
	std::thread infer_ocr_t(&infer_ocr_thread, std::ref(g_produce), std::cref(config));
	std::thread run_t(&run_thread, std::ref(g_produce), std::cref(config));
	std::thread stream_t(&stream_thread, std::ref(g_produce));

	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);

	while (!g_exit)
	{
		usleep(10000);
	}

	g_exit = true;
	produce_t.join();
	infer_det_t.join();
	infer_seg_t.join();
	infer_ocr_t.join();
	run_t.join();
	stream_t.join();

	close_shm();
	return 0;
}
