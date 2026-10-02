#pragma once

#include <chrono>

// Host timers behind --timing. A call site adds its elapsed time to a named
// accumulator; the loaders, pathtraceInit and the headless loop do this
// whether or not the flag is set (a few steady_clock reads), and timingReport
// prints the accumulators when it is. The per-bounce GPU timing in
// pathtrace.cu records only when timingEnabled().
//
// A line reads "TIMING,load.image_decode,412.3": a name and a value, in
// milliseconds unless the name says otherwise (_bytes, _mb). The sweep script
// under profiling/ parses them.

void setTiming(bool enabled);
bool timingEnabled();

// Adds value to the accumulator called name, made on first use.
void timingAdd(const char* name, double value);
// The accumulator's value so far, 0 when it does not exist.
double timingGet(const char* name);

// Adds the time from construction to destruction to name.
class TimingScope
{
public:
    explicit TimingScope(const char* name);
    ~TimingScope();
    TimingScope(const TimingScope&) = delete;
    TimingScope& operator=(const TimingScope&) = delete;

private:
    const char* name;
    std::chrono::steady_clock::time_point start;
};

// Prints every accumulator in first-use order as "TIMING,<name>,<value>".
void timingReport();
