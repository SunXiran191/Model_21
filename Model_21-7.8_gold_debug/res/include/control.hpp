class LostLine
{
public:
	LostLine() : lost_active_(false), avg_error_(0.0f), avg_xerror_(0.0f) {}

	void reset()
	{
		error_hist_.clear();
		xerror_hist_.clear();
		lost_active_ = false;
		avg_error_ = 0.0f;
		avg_xerror_ = 0.0f;
	}

	void lost_line_update(float error, float xerror, int track_side, float &out_error, float &out_xerror)
	{
		if (track_side != 0)
		{
			if (!lost_active_)
			{
				avg_error_ = average(error_hist_);
				avg_xerror_ = average(xerror_hist_);
				lost_active_ = true;
			}
			// track_side=-1(车偏左侧) → +80 右转回正; track_side=1(车偏右侧) → -80 左转回正
			out_error = track_side * 80.0f;
			out_xerror = avg_xerror_;
			return;
		}

		if (lost_active_)
		{
			reset();
		}

		push_window(error_hist_, error);
		push_window(xerror_hist_, xerror);
		out_error = error;
		out_xerror = xerror;
	}


private:
	static const std::size_t kWindowSize = 8;

	std::deque<float> error_hist_;
	std::deque<float> xerror_hist_;
	bool lost_active_;
	float avg_error_;
	float avg_xerror_;

	static void push_window(std::deque<float> &buf, float value)
	{
		buf.push_back(value);
		if (buf.size() > kWindowSize)
		{
			buf.pop_front();
		}
	}

	static float average(const std::deque<float> &buf)
	{
		if (buf.empty())
		{
			return 0.0f;
		}
		float sum = 0.0f;
		for (std::size_t i = 0; i < buf.size(); ++i)
		{
			sum += buf[i];
		}
		return sum / static_cast<float>(buf.size());
	}
};

