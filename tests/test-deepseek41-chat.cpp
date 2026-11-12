#include "chat.h"
#include <fstream>
#include <iostream>
#include <sstream>
int main() {
    std::ifstream file("models/templates/deepseek-ai-DeepSeek-V4.1-Flash-1017.jinja"); std::ostringstream ss; ss << file.rdbuf();
    auto tmpls = common_chat_templates_init(nullptr, ss.str());
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(R"([{"role":"system","content":"s","tools":[{"type":"function","namespace":"ns","function":{"name":"get_time","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]},{"role":"user","content":"q"}])"));
    inputs.add_generation_prompt = true; inputs.chat_template_kwargs["thinking_mode"] = "\"chat\"";
    inputs.enable_thinking = false; inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    common_chat_session session(tmpls.get(),nullptr,inputs,{});
    const auto & msg=session.finish(common_chat_input(std::string("\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"ns::get_time\">\n<｜DSML｜ parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>")));
    if(msg.tool_calls.size()!=1 || msg.tool_calls[0].name!="ns::get_time" || msg.tool_calls[0].arguments!="{\"city\":\"Tokyo\"}") { std::cerr << "wrong result\n"; return 1; }
    std::cout << "namespaced tool parse passed\n"; return 0;
}
