#include "ocr_api.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>

BaiduApi::BaiduApi()
    : enabled_(false)
{
    // 千帆 v2 API — 支持 VL 视觉模型
    // 环境变量优先，否则使用硬编码 IAM Access Key
    const char *env_key = std::getenv("BAIDU_API_KEY");

    const char *key = (env_key && std::strlen(env_key) > 0)
                          ? env_key
                          : "bce-v3/****-encrypted:U2FsdGVkX1+Z7H3Kj9mQwR5tVnB4cXy8pL6oI2uN0fA=";

    set_credentials(key);
}

BaiduApi::~BaiduApi() {}

void BaiduApi::set_credentials(const std::string &api_key)
{
    api_key_ = api_key;
    enabled_ = (!api_key_.empty() && api_key_ != "your_api_key_here");
}

// ============================================================
//  HTTP POST
// ============================================================
std::string BaiduApi::http_post(const std::string &url,
                                const std::string &body,
                                const std::string &bearer_token,
                                int timeout_sec)
{
    // 解析 URL: https://host:port/path
    std::string host, path;
    int port = 443;
    bool use_ssl = false;

    size_t scheme_end = url.find("://");
    size_t host_start = (scheme_end != std::string::npos) ? scheme_end + 3 : 0;
    if (url.compare(0, scheme_end, "https") == 0)
        use_ssl = true;

    size_t path_start = url.find('/', host_start);
    if (path_start != std::string::npos)
    {
        host = url.substr(host_start, path_start - host_start);
        path = url.substr(path_start);
    }
    else
    {
        host = url.substr(host_start);
        path = "/";
    }

    // 分离端口
    size_t colon = host.find(':');
    if (colon != std::string::npos)
    {
        port = std::atoi(host.substr(colon + 1).c_str());
        host = host.substr(0, colon);
    }
    else
    {
        port = use_ssl ? 443 : 80;
    }

    // 使用 HTTPS 时通过 popen curl 调用
    if (use_ssl)
    {
        // 写入唯一临时文件（PID+计数器避免冲突），fsync确保落盘
        static int s_req_id = 0;
        char tmpf[64];
        snprintf(tmpf, sizeof(tmpf), "/tmp/ocr_body_%d_%d.json", getpid(), ++s_req_id);
        FILE *fp = fopen(tmpf, "w");
        if (!fp)
            return "";
        fwrite(body.c_str(), 1, body.size(), fp);
        fflush(fp);
        fsync(fileno(fp));
        fclose(fp);

        std::ostringstream cmd;
        cmd << "curl -s --max-time " << timeout_sec
            << " -X POST '" << url << "'"
            << " -H 'Content-Type: application/json'"
            << " -H 'Authorization: Bearer " << bearer_token << "'"
            << " -d @" << tmpf << " 2>&1";

        FILE *pipe = popen(cmd.str().c_str(), "r");
        if (!pipe)
        {
            unlink(tmpf);
            return "";
        }

        std::string result;
        char buf[4096];
        while (fgets(buf, sizeof(buf), pipe))
            result += buf;
        int rc = pclose(pipe);
        if (rc != 0)
            std::cerr << "[http_post] curl exit=" << rc << std::endl;
        unlink(tmpf); // 清理临时文件
        return result;
    }

    // HTTP (非 SSL) 纯 socket 实现（备用）
    return "";
}

// ============================================================
//  语义导航 (千帆 v2 OpenAI 兼容接口)
// ============================================================
std::string BaiduApi::navigate(const std::string &ocr_text)
{
    if (ocr_text.empty())
        return "右转";

    // 无 API Key → 规则兜底
    if (!enabled_)
    {
        std::cout << "[BaiduApi] API disabled, using rule-based" << std::endl;
        return rule_based(ocr_text);
    }

    // 千帆 v2 端点: 直接 Bearer Auth，无需 token 交换
    const char *url = "https://qianfan.baidubce.com/v2/chat/completions";

    // 构建 OpenAI 兼容请求体
    std::ostringstream body;
    body << "{"
         << "\"model\":\"ernie-4.5-turbo-32k\","
         << "\"messages\":["
         << "{\"role\":\"system\",\"content\":\""
         << "你是智能驾驶导航助手。你面前只有直行（左道）和岔路（右道）两条道路。"
         << "根据道路指示牌文字判断直行或右转。"
         << "注意：OCR可能有错字（如道误为造），请根据语义推理。"
         << "规则：左侧/左道封闭或施工→走右道→右转；右侧/右道封闭或施工→走左道→直行。"
         << "如果文字模糊无法判断，请返回\\\"无法判断\\\"。"
         << "仅返回\\\"直行\\\"、\\\"右转\\\"或\\\"无法判断\\\"三个词中的一个。\"},"
         << "{\"role\":\"user\",\"content\":\"指示牌文字：" << ocr_text << "\"}"
         << "],"
         << "\"temperature\":0.1"
         << "}";

    std::string resp = http_post(url, body.str(), api_key_, 15);
    if (resp.empty())
    {
        std::cerr << "[BaiduApi !!error!!] API timeout, using rule-based" << std::endl;
        return rule_based(ocr_text);
    }

    // 解析 OpenAI 格式响应: choices[0].message.content
    size_t pos = resp.find("\"content\"");
    if (pos != std::string::npos)
    {
        pos = resp.find('"', pos + 10);
        if (pos != std::string::npos)
        {
            size_t end = resp.find('"', pos + 1);
            if (end != std::string::npos)
            {
                std::string answer = resp.substr(pos + 1, end - pos - 1);
                std::cout << "[BaiduApi] LLM answer: '" << answer << "'" << std::endl;

                // 无法判断 → 返回特殊值
                if (answer.find("无法判断") != std::string::npos)
                    return "无法判断";
                // 右转优先（避免"无法直行"等误导）
                if (answer.find("右转") != std::string::npos)
                    return "右转";
                if (answer.find("直行") != std::string::npos)
                    return "直行";
            }
        }
    }

    // API 返回错误 → 规则兜底
    if (resp.find("\"error\"") != std::string::npos)
    {
        std::cerr << "[BaiduApi !!error!!] API error, using rule-based" << std::endl;
        return rule_based(ocr_text);
    }

    return "直行"; // 兜底
}

// ============================================================
//  视觉导航 (千帆 v2 Vision API — 直接看图推理)
// ============================================================
std::string BaiduApi::navigate_vision(const std::string &image_base64,
                                      const std::string &model_name)
{
    if (image_base64.empty())
        return "右转";

    if (!enabled_)
    {
        std::cout << "[BaiduApi Vision] API disabled, fallback RIGHT" << std::endl;
        return "右转";
    }

    const char *url = "https://qianfan.baidubce.com/v2/chat/completions";

    std::ostringstream body;
    body << "{"
         << "\"model\":\"" << model_name << "\","
         << "\"messages\":[{"
         << "\"role\":\"user\","
         << "\"content\":["
         << "{\"type\":\"text\",\"text\":\""
         << "你是智能驾驶导航助手。你面前只有直行（左道/直道）和右道（岔路）两条路。"
         << "根据道路指示牌文字判断直行或右转。"
         << "示例：右侧是一条不归路→走左道→直行；左侧道路崎岖，右侧一马平川→走右道→右转"
         << "请直接输出决策"
         << "\"},"
         << "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,"
         << image_base64 << "\"}}"
         << "]"
         << "}],"
         << "\"temperature\":0.01"
         << "}";

    std::string body_str = body.str();
    std::cerr << "[BaiduApi Vision] body size: " << body_str.size() << " bytes" << std::endl;

    std::string resp = http_post(url, body_str, api_key_, 30);
    if (resp.empty())
    {
        std::cerr << "[BaiduApi Vision !!error!!] HTTP request failed (empty response)" << std::endl;
        return "右转";
    }

    // 输出完整响应用于调试
    std::cout << "[BaiduApi Vision] raw response: " << resp.substr(0, 400) << std::endl;

    size_t pos = resp.find("\"content\"");
    if (pos != std::string::npos)
    {
        pos = resp.find('"', pos + 10);
        if (pos != std::string::npos)
        {
            size_t end = resp.find('"', pos + 1);
            if (end != std::string::npos)
            {
                std::string answer = resp.substr(pos + 1, end - pos - 1);
                std::cout << "[BaiduApi Vision] answer: '" << answer << "'" << std::endl;

                // 优先解析 "决策=" 或 "快策=" 部分
                size_t dec_pos = answer.find("决策=");
                if (dec_pos == std::string::npos)
                    dec_pos = answer.find("快策="); // 模型偶尔typo
                std::string decision_part = (dec_pos != std::string::npos)
                                                ? answer.substr(dec_pos)
                                                : answer;

                // 只看决策=后面的关键词（避免路牌文字中"右道""直道"干扰）
                if (decision_part.find("右转") != std::string::npos)
                    return "右转";
                if (decision_part.find("直行") != std::string::npos)
                    return "直行";
                if (decision_part.find("无法判断") != std::string::npos)
                    return "无法判断";

                // 兜底：全文本搜索（右转优先避免"无法直行"误导）
                if (answer.find("右转") != std::string::npos)
                    return "右转";
                if (answer.find("直行") != std::string::npos)
                    return "直行";
            }
        }
    }

    if (resp.find("\"error\"") != std::string::npos)
    {
        std::cerr << "[BaiduApi Vision !!error!!] API error" << std::endl;
        return "右转";
    }

    return "右转"; // 安全兜底
}

// ============================================================
//  无 API / API 失败时的规则兜底
// ============================================================
std::string BaiduApi::rule_based(const std::string &ocr_text)
{
    // ── 语义推理：左侧/左道封闭 → 走右侧 → 右转 ──
    // 包含 OCR 常见误识别变体（如 "道"→"造"、"闭"→"闲" 等）
    const char *left_block_kw[] = {
        "左侧道路崎岖，右侧一马平川", "左侧封闭", "左道施工", "左侧施工",
        "左道关闭", "左方封闭", "左方施工",
        "左造施工", "左造封闭", // OCR 误识别变体
        "左倒施工", "左道施工"};
    for (size_t i = 0; i < sizeof(left_block_kw) / sizeof(left_block_kw[0]); ++i)
    {
        if (ocr_text.find(left_block_kw[i]) != std::string::npos)
            return "右转";
    }

    // ── 右侧/右道封闭 → 走左侧 → 直行 ──
    const char *right_block_kw[] = {
        "右侧是一条不归路", "右侧封闭", "右道施工", "右侧施工",
        "右道关闭", "右方封闭", "右方施工"};
    for (size_t i = 0; i < sizeof(right_block_kw) / sizeof(right_block_kw[0]); ++i)
    {
        if (ocr_text.find(right_block_kw[i]) != std::string::npos)
            return "直行";
    }

    // 右转关键词（优先级更高）
    const char *right_kw[] = {"右转", "右侧", "右", "绕行", "转入", "拐入"};
    for (size_t i = 0; i < sizeof(right_kw) / sizeof(right_kw[0]); ++i)
    {
        if (ocr_text.find(right_kw[i]) != std::string::npos)
            return "右转";
    }

    // 直行关键词
    const char *straight_kw[] = {"直行", "直", "前", "继", "通过", "通行"};
    for (size_t i = 0; i < sizeof(straight_kw) / sizeof(straight_kw[0]); ++i)
    {
        if (ocr_text.find(straight_kw[i]) != std::string::npos)
            return "直行";
    }

    return "直行"; // 默认直行
}
