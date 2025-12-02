#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <Eigen/Dense>
#include <vector>
#include <memory>

#include "rho_filter/rhoFilter.hpp"

using std::placeholders::_1;

class RhoFilterNode : public rclcpp::Node {
public:
    RhoFilterNode() : Node("rho_filter_node") {
        declare_parameter("sampling_time_seconds", 0.001);
        declare_parameter("state_space_dim", 1);
        declare_parameter("alpha", 1.0);
        declare_parameter("k1", 1.0);
        declare_parameter("k2", 1.0);
        declare_parameter("k3", 1.0);

        double ts = get_parameter("sampling_time_seconds").as_double();
        int state_space_dim = get_parameter("state_space_dim").as_int();
        double alpha = get_parameter("alpha").as_double();
        double k1 = get_parameter("k1").as_double();
        double k2 = get_parameter("k2").as_double();
        double k3 = get_parameter("k3").as_double();

        filter_ = std::make_unique<rhoFilter>(ts, state_space_dim, alpha, k1, k2, k3);

        pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("filter_output", 10);
        sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            "filter_input", 10, std::bind(&RhoFilterNode::topic_callback, this, _1));
    }

private:
    void topic_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        int state_space_dim = get_parameter("state_space_dim").as_int();

        if (msg->data.size() != static_cast<size_t>(state_space_dim)) {
            RCLCPP_ERROR(get_logger(), "Dim mismatch. Expected %d, got %zu", state_space_dim, msg->data.size());
            return;
        }

        Eigen::MatrixXd input_pos(state_space_dim, 1);
        for (int i = 0; i < state_space_dim; ++i) {
            input_pos(i, 0) = msg->data[i];
        }

        filter_->propagate_filter(input_pos);

        // zeta is 4n size. We want the last n elements (estimated state derivative).
        Eigen::MatrixXd est_state_derivative = filter_->zeta.bottomRows(state_space_dim);

        std_msgs::msg::Float64MultiArray out_msg;
        out_msg.layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
        out_msg.layout.dim[0].size = state_space_dim;
        out_msg.layout.dim[0].label = "estimated_derivative";
        out_msg.data.resize(state_space_dim);

        for (int i = 0; i < dim; ++i) {
            out_msg.data[i] = est_state_derivative(i, 0);
        }

        pub_->publish(out_msg);
    }

    std::unique_ptr<rhoFilter> filter_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr sub_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RhoFilterNode>());
    rclcpp::shutdown();
    return 0;
}